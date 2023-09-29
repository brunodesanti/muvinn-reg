"""
Author: Bruno De Santi, PhD
Affiliation: Multi-modality Medical Imaging Lab (M3I Lab), University of Twente, Enschede, The Netherlands
Date: 20/09/2023

Description: Normalized cross correlation for network optimization

Acknowledge: The code was adapted from
1) https://github.com/BDdeVos/TorchIR
2) J. M. Wolterink, J. C. Zwienenberg, and C. Brune, “Implicit Neural Representations for Deformable Image Registration,” in Proceedings of Machine Learning Research (2022) [url:https://openreview.net/forum?id=BP29eKzQBu3]
GitHub: https://github.com/MIAGroupUT/IDIR

Paper/Project Title: Automated three-dimensional image registration for longitudinal photoacoustic imaging (De Santi et al. 2023, JBO)
Paper/Project URL: https://github.com/brunodesanti/muvinn-reg

License: [Specify the license, e.g., MIT, GPL, etc.]
"""
import math
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

def ncc(x1, x2, e=1e-10):
    assert x1.shape == x2.shape, "Inputs are not of similar shape"
    x1 = x1.view(-1, 5**3)
    x2 = x2.view(-1, 5**3)
    cc = ((x1 - x1.mean(dim=1)[:, None]) * (x2 - x2.mean(dim=1)[:, None])).mean(dim=1)
    std = x1.std(dim=1) * x2.std(dim=1)
    ncc = torch.mean(cc/(std+e))
    return ncc
    
class NCC(_Loss):
    def __init__(self, use_mask: bool = False):
        super().__init__()
        self.forward = self.metric

    def metric(self, fixed: Tensor, warped: Tensor) -> Tensor:
        return -ncc(fixed, warped)