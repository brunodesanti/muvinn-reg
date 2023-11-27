"""
Author: Bruno De Santi, PhD
Affiliation: Multi-modality Medical Imaging Lab (M3I Lab), University of Twente, Enschede, The Netherlands
Date: 20/09/2023

Description: Collection of processing techniques used in MUVINN-reg

Paper/Project Title: Automated three-dimensional image registration for longitudinal photoacoustic imaging (De Santi et al. 2023, JBO)
Paper/Project URL: https://github.com/brunodesanti/muvinn-reg

License: [Specify the license, e.g., MIT, GPL, etc.]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
import SimpleITK as sitk

# Crop image according to bounding box
def crop_rec(rec, image_crop_region):
    rec = rec[
        image_crop_region[0][0]:image_crop_region[0][1],
        image_crop_region[1][0]:image_crop_region[1][1],
        image_crop_region[2][0]:image_crop_region[2][1],
        ]
        
    return rec

# Pad image
def pad_rec(rec, padding):
    rec = np.pad(rec,(
    (padding[0][0], padding[0][1]),
    (padding[1][0], padding[1][1]),
    (padding[2][0],padding[2][1])
    ),'constant')
    
    return rec
    
# Scale image intensities
def scale_intensity(rec, a_min = 0, a_max = 800, b_min = 0, b_max = 1, clip = True):
    ScaleIntensity = monai.transforms.ScaleIntensityRange(a_min, a_max, b_min, b_max, clip)
    rec = ScaleIntensity(rec).numpy()
    
    return rec

# Gaussian smoothing filter on GPU
def gaussian_smooth(image, sigma, gpu = 'cuda'):
    with torch.no_grad():
        ks = 6 * sigma
        x = np.arange(-int(ks / 2),int(ks / 2) + 1, 1)
        xx, yy, zz = np.meshgrid(x, x, x)
        w = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2)/(2 * sigma ** 2))
        w /= w.sum() 
        
        g_t = torch.tensor(w, device = gpu).float()
        pad_image = F.pad(image.unsqueeze(0), pad = (int(ks / 2),)*6 , mode = 'reflect').squeeze(0)
        output = F.conv3d(pad_image.unsqueeze(0).unsqueeze(0), g_t.unsqueeze(0).unsqueeze(0), padding = 'valid').squeeze()
        
    return output

# Compute fast Frangi vesselness filtering on CUDA enabled PyTorch
# Acknowledgment: The author took inspiration and adapted from the following:
# Dirk-Jan Kroon, Hessian based Frangi Vesselness filter (https://www.mathworks.com/matlabcentral/fileexchange/24409-hessian-based-frangi-vesselness-filter)
def frangi_cuda(image, options = None, gpu = 'cuda'):
    if options is None:
        options = dict()
        options['sigma'] = 2
        options['alpha'] = 0.5
        options['beta'] = 0.5
        options['gamma'] = 0.05
        options['bw_flag'] = True
       
    with torch.no_grad():
        IF = gaussian_smooth(image, options['sigma'], gpu = gpu)

        Dx = torch.gradient(IF, spacing = 1, dim = 1, edge_order = 2)[0]
        Dy = torch.gradient(IF, spacing = 1, dim = 0, edge_order = 2)[0]
        Dz = torch.gradient(IF, spacing = 1, dim = 2, edge_order = 2)[0]

        Dzz = torch.gradient(Dz, spacing = 1, dim = 2, edge_order = 2)[0]

        Dyy = torch.gradient(Dy, spacing = 1, dim = 0, edge_order = 2)[0]
        Dyz = torch.gradient(Dy, spacing = 1, dim = 2, edge_order = 2)[0]

        Dxx = torch.gradient(Dx, spacing = 1, dim = 1, edge_order = 2)[0]
        Dxy = torch.gradient(Dx, spacing = 1, dim = 0, edge_order = 2)[0]
        Dxz = torch.gradient(Dx, spacing = 1, dim = 2, edge_order = 2)[0]

        # Correct for scaling
        c = options['sigma']**2
        Dxx = c * Dxx
        Dxy = c * Dxy
        Dxz = c * Dxz
        Dyy = c * Dyy
        Dyz = c * Dyz
        Dzz = c * Dzz

        D = torch.zeros(torch.numel(image), 3, 3, device = gpu)

        D[:,0,0] = Dxx.view(-1)
        D[:,1,1] = Dyy.view(-1)
        D[:,2,2] = Dzz.view(-1)

        D[:,0,1] = Dxy.view(-1)
        D[:,0,2] = Dxz.view(-1)
        D[:,1,0] = Dxy.view(-1)
        D[:,2,0] = Dxz.view(-1)

        D[:,1,2] = Dyz.view(-1)
        D[:,2,1] = Dyz.view(-1)

        veigh = torch.vmap(torch.linalg.eigvalsh)
        
        batch_size = 100000; # One might lower this depending on GPU memory available
        index = 0

        SV = torch.zeros(torch.numel(image), 3, device = gpu)
        for D_batch in torch.split(D, batch_size):  
            SV[index:index + batch_size,:] =  veigh(D_batch)
            index = index + batch_size
        indices = torch.argsort(torch.abs(SV), descending = False, dim = 1)
        SV = torch.gather(SV, dim = 1, index = indices)
        aSV = torch.abs(SV)

        Ra = torch.divide(aSV[:,1], aSV[:,2])
        Rb = torch.divide(aSV[:,0], torch.sqrt(torch.mul(aSV[:,1], aSV[:,2])))
        S = torch.norm(aSV, p = 'fro', dim = 1)

        expRa = 1 - torch.exp(torch.divide(-Ra ** 2., 2 * options['alpha'] ** 2))
        expRb = torch.exp(torch.divide(-Rb ** 2., 2 * options['beta'] ** 2))
        expS  = 1 - torch.exp(torch.divide(-S ** 2., 2 * options['gamma'] ** 2))

        VR = torch.mul(torch.mul(expRa, expRb),expS); 

        if options['bw_flag']:
            VR[torch.where(SV[:,1] > 0)] = 0
            VR[torch.where(SV[:,2] > 0)] = 0
        else:
            VR[torch.where(SV[:,1] < 0)] = 0
            VR[torch.where(SV[:,2] < 0)] = 0
            
    return torch.reshape(torch.nan_to_num(VR), image.shape)

# Adaptive intensity modulation
# Acknowledgment: The author has taken inspiration and adapted from the following:
# L. Lin et al., “High-speed three-dimensional photoacoustic computed tomography for preclinical research and clinical translation,” Nat Commun, vol. 12, no. 1, pp. 1–10, Feb. 2021, doi: 10.1038/s41467-021-21232-1.
def aim_cuda(hessian, options = None, gpu = 'cuda'):
    if options is None:
        options = dict()
        options['half_size_win'] = 5
        options['min_sd'] = 0.2
        
    with torch.no_grad():
        w = np.ones((options['half_size_win'] * 2 + 1,
                    options['half_size_win'] * 2 + 1,
                    options['half_size_win'] * 2 + 1),
                    dtype = 'float')
        w /= w.sum() 
        
        g_t = torch.tensor(w, device = gpu).float()
        pad_hessian = F.pad(hessian.unsqueeze(0), pad = (int(options['half_size_win']),)*6, mode = 'reflect').squeeze(0)
        M = F.conv3d(pad_hessian.unsqueeze(0).unsqueeze(0), g_t.unsqueeze(0).unsqueeze(0),padding = 'valid').squeeze()

        D = (hessian - M) ** 2
        pad_D = F.pad(D.unsqueeze(0), pad = (int(options['half_size_win']),)*6, mode = 'reflect').squeeze(0)
        V = F.conv3d(pad_D.unsqueeze(0).unsqueeze(0), g_t.unsqueeze(0).unsqueeze(0),padding = 'valid').squeeze()

        SD = torch.sqrt(V)
        SD = torch.divide(SD, torch.max(SD))
        SD[SD < options['min_sd']] = options['min_sd']
        SD = 1 / SD
        
        hessian = hessian * SD
        hessian = torch.divide(hessian, torch.max(hessian))
        
    return hessian
    
def frangi_aim(rec, frangi_options, aim_options, gpu = 'cuda'):
    output = aim_cuda(frangi_cuda(rec, frangi_options, gpu = 'cuda'), aim_options, gpu = 'cuda')
    
    return output
    
# PAM3 image preprocessing for visualization
# Acknowledgments:
# 1) L. Lin et al., “High-speed three-dimensional photoacoustic computed tomography for preclinical research and clinical translation,” Nat Commun, vol. 12, no. 1, pp. 1–10, Feb. 2021, doi: 10.1038/s41467-021-21232-1.
# 2) Dirk-Jan Kroon, Hessian based Frangi Vesselness filter (https://www.mathworks.com/matlabcentral/fileexchange/24409-hessian-based-frangi-vesselness-filter)
def processing_vis(rec, frangi_options = None, aim_options = None, gpu = 'cuda'):
    if frangi_options is None:
        frangi_options = dict()
        frangi_options['sigmas'] = (1.5, 2, 2.5)
        frangi_options['alpha'] = 0.5
        frangi_options['beta'] = 0.5
        frangi_options['gamma'] = 0.025
        frangi_options['bw_flag'] = True  
        
    if aim_options is None:
        aim_options = dict()
        aim_options['half_size_win'] = 5
        aim_options['min_sd'] = 0.1
        aim_options['weights'] = (0.5, 0.5)
    
    rec = torch.FloatTensor(rec).to(gpu)
    hessian = torch.zeros_like(rec)
    
    for sigma in frangi_options['sigmas']:
        frangi_options['sigma'] = sigma
        hessian = torch.maximum(hessian, frangi_cuda(rec, frangi_options, gpu = 'cuda'))
    hessian_mod = aim_cuda(hessian, aim_options, gpu = 'cuda')
    
    output = aim_options['weights'][0] * rec + aim_options['weights'][1] * hessian_mod
    output[output < 0] = 0
    
    return hessian.cpu().detach().numpy(), hessian_mod.cpu().detach().numpy(), output.cpu().detach().numpy()
    
# Vessel segmentation by adaptive thresholding
def segment_vessels(rec, depth_map, ti = 0.06, tf = 0.01, tau = 100):
    thresh_map = tf + (ti - tf) * np.exp(-tau * depth_map)
    vessel_seg = rec > thresh_map
    return vessel_seg