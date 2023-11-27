"""
Author: Bruno De Santi, PhD
Affiliation: Multi-modality Medical Imaging Lab (M3I Lab), University of Twente, Enschede, The Netherlands
Date: 20/09/2023

Description: Collection of visualizing techniques used in MUVINN-reg

Paper/Project Title: Automated three-dimensional image registration for longitudinal photoacoustic imaging (De Santi et al. 2023, JBO)
Paper/Project URL: https://github.com/brunodesanti/muvinn-reg

License: [Specify the license, e.g., MIT, GPL, etc.]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Compute RGB overlays of two images 
def comp_rgb_overlay(mov_ch, fix_ch, alpha = 0.5):
    fix_ch = fix_ch / np.max(fix_ch)
    mov_ch = mov_ch / np.max(mov_ch)
    
    colors1 = [(1, 1, 1), (1, 0, 0)] # first color is white, last is red
    cmap1 = LinearSegmentedColormap.from_list(
        "Custom", colors1, N = 256)
    colors2 = [(1, 1, 1), (0, 0, 1)] # first color is white, last is red
    cmap2 = LinearSegmentedColormap.from_list(
        "Custom", colors2, N = 256)
        
    mov_ch_rgb = plt.get_cmap(cmap1)(mov_ch)
    fix_ch_rgb = plt.get_cmap(cmap2)(fix_ch)
 
    overlay_rgb = alpha * mov_ch_rgb + (1-alpha) * fix_ch_rgb
    
    return(overlay_rgb)

# Plot maximum intensity projections in coronal, sagittal and axial views
def plot_mips(rec, mask = None, cmap = 'gray', vmin = 0, vmax = 1):
    rec_shape = rec.shape
    
    fig = plt.figure(figsize = (19.8, 19.8))

    a1 = plt.subplot(2, 2, 1)
    plt.imshow(np.max(rec, axis = 2), cmap = cmap)
    if mask is not None:
        plt.imshow(np.max(mask, axis = 2), cmap = 'turbo', alpha = 0.5)
    plt.axis('off')
 
    a2 = plt.subplot(2, 2, 2)
    plt.imshow(np.flip(np.max(rec, axis = 1),axis = 1), cmap = cmap)
    if mask is not None:
        plt.imshow(np.flip(np.max(mask, axis = 1),axis = 1), cmap = 'turbo', alpha = 0.5)
    plt.axis('off')
    
    a3 = plt.subplot(2, 2, 3)
    plt.imshow(np.flip(np.max(rec, axis = 0).T, axis = 0), cmap = cmap)
    if mask is not None:
        plt.imshow(np.flip(np.max(mask, axis = 0).T, axis = 0), cmap = 'turbo', alpha = 0.5)
    plt.axis('off')
    
    return fig
    
# Plot maximum intensity projections in coronal, sagittal and axial views of RGB overlays
def plot_aligned_mips(cor_mov_ch, cor_fix_ch, alpha = 0.5):
    cor_fix_mip = np.max(cor_fix_ch,axis=2)
    sag_fix_mip = np.flip(np.max(cor_fix_ch, axis = 1),axis = 1)
    axi_fix_mip = np.flip(np.max(cor_fix_ch, axis = 0).T, axis = 0)
    
    cor_mov_mip = np.max(cor_mov_ch, axis = 2)
    sag_mov_mip = np.flip(np.max(cor_mov_ch, axis = 1), axis = 1)
    axi_mov_mip = np.flip(np.max(cor_mov_ch, axis = 0).T, axis = 0)
    
    fig = plt.figure(figsize = (19.8, 19.8))

    a1 = plt.subplot(2, 2, 1)
    plt.imshow(comp_rgb_overlay(cor_mov_mip, cor_fix_mip,alpha))
    plt.axis('off')

    a2 = plt.subplot(2, 2, 2)
    plt.imshow(comp_rgb_overlay(sag_mov_mip, sag_fix_mip,alpha))
    plt.axis('off')

    a3 = plt.subplot(2, 2, 3)
    plt.imshow(comp_rgb_overlay(axi_mov_mip, axi_fix_mip,alpha))
    plt.axis('off')
    
    return fig
    
# Plot maximum intensity projections in coronal, sagittal and axial views of RGB overlays at different depths
def plot_aligned_mips_depth(moving_image, fixed_image, depth_map, depths = (6, 4, 2, 0), alpha = 0.5):
    figs = list()
    for depth in depths:
        fixed_layer = np.zeros_like(fixed_image)
        moving_layer = np.zeros_like(moving_image)

        fixed_layer[np.where(100 * depth_map > depth)] =  fixed_image[np.where(100 * depth_map > depth)]
        moving_layer[np.where(100 * depth_map > depth)] =  moving_image[np.where(100 * depth_map > depth)]
        
        figs.append(plot_aligned_mips(moving_layer, fixed_layer, alpha))
        
    return figs