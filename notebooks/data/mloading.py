"""
Author: Bruno De Santi, PhD
Affiliation: Multi-modality Medical Imaging Lab (M3I Lab), University of Twente, Enschede, The Netherlands
Date: 20/09/2023

Description: Functions for loading reconstructions and cup masks (.mat)

Paper/Project Title: Automated three-dimensional image registration for longitudinal photoacoustic imaging (De Santi et al. 2023, JBO)
Paper/Project URL: https://github.com/brunodesanti/muvinn-reg

License: [Specify the license, e.g., MIT, GPL, etc.]
"""

import os
import numpy as np
import h5py
import scipy.io as sio 

def load_PAM3_rec(rec_path):
    data = h5py.File(rec_path,'r')
    rec = np.swapaxes(np.array(data['p0_rec']), 0, 2)
    rec = np.flip(rec, 0)
    return rec
    
def load_PAM3_cup_mask(omask_path, size):
    mask_path = omask_path + os.path.sep + 'cup_mask_{}.mat'.format(size)
    cup_mask = sio.loadmat(mask_path)['cup_mask']
    
    depth_path = omask_path + os.path.sep + 'depth_map_{}.mat'.format(size)
    depth_map = sio.loadmat(depth_path)['depth_map']
    return cup_mask, depth_map
