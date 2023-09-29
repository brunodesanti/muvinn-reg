"""
Author: Bruno De Santi, PhD
Affiliation: Multi-modality Medical Imaging Lab (M3I Lab), University of Twente, Enschede, The Netherlands
Date: 20/09/2023

Description: Collection of metrics to evaluate registration performance
MSE: mean square error
PSNR: peak signal-to-noise ratio
SSIM: structural similarity index
NCC: normalized cross correlation
TRE: target registration error, distance between corresponding points
DICE: Dice similarity coefficient between segmented structure

Paper/Project Title: Automated three-dimensional image registration for longitudinal photoacoustic imaging (De Santi et al. 2023, JBO)
Paper/Project URL: https://github.com/brunodesanti/muvinn-reg

License: [Specify the license, e.g., MIT, GPL, etc.]
"""

import numpy as np
import skimage.metrics as skme
import torch
import os
from string import digits

# Compute similarity metrics between pair of images. Optional "masks" for DICE and optional "landmarks" for TRE.
def similarity(images = None, masks = None, landmarks = None):
    metrics = dict.fromkeys(['mse', 'psnr', 'ssim', 'ncc', 'tre', 'dice'])

    metrics['mse'] = skme.mean_squared_error(images['fixed'], images['moving'])
    
    metrics['psnr'] = skme.peak_signal_noise_ratio(images['fixed'], images['moving'], data_range = 1)
    
    metrics['ssim'] = skme.structural_similarity(images['fixed'], images['moving'], data_range = 1, gaussian_weights = True, sigma = 9, use_sample_covariance = True)

    if masks is not None:
        intersection = np.logical_and(np.bool_(masks['fixed']).flatten(), np.bool_(masks['moving']).flatten())
        metrics['dice'] = 2. * intersection.sum() / (masks['fixed'].sum() + masks['moving'].sum())
        
    if landmarks is not None:
        mean_tre, std_tre = compute_landmark_accuracy(landmarks['reg'], landmarks['gt'], voxel_size = [0.4, 0.4, 0.4])
        metrics['tre'] = [mean_tre, std_tre]
        
    metrics['ncc'] = float(ncc_cuda(torch.FloatTensor(images['fixed']).cuda(), torch.FloatTensor(images['moving']).cuda()))
   
    return metrics

# Compute normalized cross-correlation
def ncc_cuda(x1, x2):
    e = 1e-10
    with torch.no_grad():
        x1 = x1.view(-1)
        x2 = x2.view(-1)
        cc = ((x1 - x1.mean(dim=0)) * (x2 - x2.mean(dim=0))).mean(dim=0)
        std = x1.std(dim=0) * x2.std(dim=0)
        ncc = torch.mean(cc/(std+e))
        
    return ncc.cpu().detach().numpy()
    
# Compute mean and standard deviation of distances between corresponding points
def compute_landmark_accuracy(landmarks_pred, landmarks_gt, voxel_size):
    landmarks_pred = np.round(landmarks_pred)
    landmarks_gt = np.round(landmarks_gt)

    difference = landmarks_pred - landmarks_gt
    difference = np.abs(difference)
    difference = difference * voxel_size
    difference = np.square(difference)
    difference = np.sum(difference, 1)
    difference = np.sqrt(difference)

    mean = np.mean(difference)
    std = np.std(difference)

    mean = np.round(mean, 2)
    std = np.round(std, 2)

    return mean, std

# Load landmarks from MEVIS MarkerList format files
def load_landmarks(path, fixed_id, moving_id):
    with open(path + os.path.sep + '{}-{}.txt'.format(fixed_id, moving_id), 'r') as file:
        landmarks = file.read()
        
    fixed_landmarks = from_string_list_to_np_array(landmarks.split('\n')[0])
    moving_landmarks = from_string_list_to_np_array(landmarks.split('\n')[1])
    
    fixed_landmarks = fixed_landmarks[:,[2,1,0]]
    moving_landmarks = moving_landmarks[:,[2,1,0]]
    
    return fixed_landmarks, moving_landmarks
    
# Convert string to numerical coordinates
def from_string_list_to_np_array(landmarks):
    list_str_x = landmarks.split(' ')[0:-1:4]
    list_str_y = landmarks.split(' ')[1:-1:4]
    list_str_z = landmarks.split(' ')[2:-1:4]
    
    array = np.zeros((len(list_str_x),3))
    
    for i, string in enumerate(list_str_x):
        array[i][0] = float(''.join(c for c in string if c in digits))
        array[i][1] = float(''.join(c for c in list_str_y[i] if c in digits))
        array[i][2] = float(''.join(c for c in list_str_z[i] if c in digits))
        
    return array
    
# Transform point coordinates using the network
def register_landmarks(network, landmarks, image_size):
    scale_of_axes = [(0.5 * s) for s in image_size]
    coordinate_tensor = torch.FloatTensor(landmarks / (scale_of_axes)) - 1.0
    output = network(coordinate_tensor.cuda())
    delta = output.cpu().detach().numpy() * (scale_of_axes)
   
    return landmarks + delta, delta
    
# Extract displacement field by forwarding through the network every coordinate of the domain
def displacement_field(network, coordinate_tensor, image_size, forward_batch_size = 10000):
    scale_of_axes = [(0.5 * s) for s in image_size]
    output = torch.zeros_like(coordinate_tensor)
    with torch.no_grad():
        index = 0
        for grid_batch in torch.split(coordinate_tensor, forward_batch_size):
            output[index:index + forward_batch_size,:] = network(grid_batch)
            index = index + forward_batch_size
        field = output.cpu().detach().numpy() * (scale_of_axes)
    
    return field.reshape(image_size  + (3,))
    