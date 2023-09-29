"""
Author: Bruno De Santi, PhD
Affiliation: Multi-modality Medical Imaging Lab (M3I Lab), University of Twente, Enschede, The Netherlands
Date: 20/09/2023

Description: "trilinear" or "nearest" interpolation used in MUVINN-reg

Acknowledge: The code was adapted from 
J. M. Wolterink, J. C. Zwienenberg, and C. Brune, “Implicit Neural Representations for Deformable Image Registration,” in Proceedings of Machine Learning Research (2022) [url:https://openreview.net/forum?id=BP29eKzQBu3]
GitHub: https://github.com/MIAGroupUT/IDIR

Paper/Project Title: Automated three-dimensional image registration for longitudinal photoacoustic imaging (De Santi et al. 2023, JBO)
Paper/Project URL: https://github.com/brunodesanti/muvinn-reg

License: [Specify the license, e.g., MIT, GPL, etc.]
"""

import torch

def fast_interpolation(input_array, x_indices, y_indices, z_indices, method='trilinear'):
    dims = input_array.shape

    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[2] - 1) * 0.5
    
    if method == 'trilinear':
        x0 = torch.floor(x_indices.detach()).to(torch.long)
        y0 = torch.floor(y_indices.detach()).to(torch.long)
        z0 = torch.floor(z_indices.detach()).to(torch.long)

        x0 = torch.clamp(x0, 0, dims[0] - 1)
        y0 = torch.clamp(y0, 0, dims[1] - 1)
        z0 = torch.clamp(z0, 0, dims[2] - 1)

        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        x1 = torch.clamp(x1, 0, dims[0] - 1)
        y1 = torch.clamp(y1, 0, dims[1] - 1)
        z1 = torch.clamp(z1, 0, dims[2] - 1)

        x = x_indices - x0
        y = y_indices - y0
        z = z_indices - z0
        
        output = (
        input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
        + input_array[x1, y0, z0] * x * (1 - y) * (1 - z)
        + input_array[x0, y1, z0] * (1 - x) * y * (1 - z)
        + input_array[x0, y0, z1] * (1 - x) * (1 - y) * z
        + input_array[x1, y0, z1] * x * (1 - y) * z
        + input_array[x0, y1, z1] * (1 - x) * y * z
        + input_array[x1, y1, z0] * x * y * (1 - z)
        + input_array[x1, y1, z1] * x * y * z
        )
        
    elif method == 'nearest':
        x0 = torch.round(x_indices.detach()).to(torch.long)
        y0 = torch.round(y_indices.detach()).to(torch.long)
        z0 = torch.round(z_indices.detach()).to(torch.long)

        x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
        y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
        z0 = torch.clamp(z0, 0, input_array.shape[2] - 1)
        
        output = (
        input_array[x0, y0, z0]
        )

    else:
        print('WARNING: ' + method + ' not recognized as interpolation method, picked "nearest" instead')
        x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
        y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5
        z_indices = (z_indices + 1) * (input_array.shape[2] - 1) * 0.5

        x0 = torch.round(x_indices.detach()).to(torch.long)
        y0 = torch.round(y_indices.detach()).to(torch.long)
        z0 = torch.round(z_indices.detach()).to(torch.long)

        x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
        y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
        z0 = torch.clamp(z0, 0, input_array.shape[2] - 1)
        
        output = (
        input_array[x0, y0, z0]
        )
    return output
