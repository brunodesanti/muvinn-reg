"""
Author: Bruno De Santi, PhD
Affiliation: Multi-modality Medical Imaging Lab (M3I Lab), University of Twente, Enschede, The Netherlands
Date: 20/09/2023

Description: Deformation field regularization functions 

Acknowledge: The code was adapted from
1) J. M. Wolterink, J. C. Zwienenberg, and C. Brune, “Implicit Neural Representations for Deformable Image Registration,” in Proceedings of Machine Learning Research (2022) [url:https://openreview.net/forum?id=BP29eKzQBu3]
GitHub: https://github.com/MIAGroupUT/IDIR
2) V. Sitzmann et al., “Implicit Neural Representations with Periodic Activation Functions,” in Proc. 508 NeurIPS, H. Larochelle et al., Eds. (2020) [arXiv:2006.09661]
GitHub: https://github.com/vsitzmann/siren

Paper/Project Title: Automated three-dimensional image registration for longitudinal photoacoustic imaging (De Santi et al. 2023, JBO)
Paper/Project URL: https://github.com/brunodesanti/muvinn-reg

License: [Specify the license, e.g., MIT, GPL, etc.]
"""

import torch
from torch import nn
import numpy as np

class Siren(nn.Module):
    """This is a dense neural network with sine activation functions.

    Arguments:
    layers -- ([*int]) amount of nodes in each layer of the network, e.g. [3, 16, 16, 3]
    gpu -- (boolean) use GPU when True, CPU when False
    weight_init -- (boolean) use special weight initialization if True
    omega -- (float) parameter used in the forward function
    """
    def __init__(self, layers, omega=30):
        """Initialize the network."""

        super(Siren, self).__init__()
        
        self.n_layers = len(layers) - 1
        self.omega = omega

        # Make the layers
        self.layers = []
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

            # Weight Initialization
            with torch.no_grad():
                if i == 0:
                    self.layers[-1].weight.uniform_(-0.5 / layers[i],
                                                    0.5 / layers[i])
                else:
                    self.layers[-1].weight.uniform_(-np.sqrt(3 / layers[i]) / self.omega,
                                                    np.sqrt(3 / layers[i]) / self.omega)

        # Combine all layers to one model
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        """The forward function of the network."""
        # Perform sinusoidal pass on all layers except for the last one
        for layer in self.layers[:-1]:
            z = layer(x)            
            x = torch.sin(self.omega * z)

        # Propagate through final layer and return the output
        x = self.layers[-1](x)    
        return x
        
class MLP(nn.Module):
    def __init__(self, layers):
        """Initialize the network."""

        super(MLP, self).__init__()
        self.n_layers = len(layers) - 1

        # Make the layers
        self.layers = []
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

        # Combine all layers to one model
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        """The forward function of the network."""

        # Perform relu on all layers except for the last one
        for layer in self.layers[:-1]:
            x = torch.nn.functional.relu(layer(x))

        # Propagate through final layer and return the output
        return self.layers[-1](x)