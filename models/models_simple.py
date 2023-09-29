"""
Author: Bruno De Santi, PhD
Affiliation: Multi-modality Medical Imaging Lab (M3I Lab), University of Twente, Enschede, The Netherlands
Date: 20/09/2023

Description: Learning model of MUVINN-reg

Acknowledge: The code was adapted from
1) J. M. Wolterink, J. C. Zwienenberg, and C. Brune, “Implicit Neural Representations for Deformable Image Registration,” in Proceedings of Machine Learning Research (2022) [url:https://openreview.net/forum?id=BP29eKzQBu3]
GitHub: https://github.com/MIAGroupUT/IDIR

Paper/Project Title: Automated three-dimensional image registration for longitudinal photoacoustic imaging (De Santi et al. 2023, JBO)
Paper/Project URL: https://github.com/brunodesanti/muvinn-reg

License: [Specify the license, e.g., MIT, GPL, etc.]
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import tqdm as tqdm
import monai
import numpy as np
import matplotlib.pyplot as plt

from utils import processing
from utils import visualizing

from networks import networks
from objectives import ncc
from objectives import regularizers
from objectives import interpolation
from monai.data import meta_tensor

ScaleIntensity = monai.transforms.ScaleIntensity(minv = 0.0, maxv = 1.0)

class ImplicitRegistrator:
    """This class contains functions which are useful for all the learning models."""

    def __call__(self, output_shape = (28, 28, 28), moving_image = None, moving_mask = None, forward_batch_size = 10000):
        """Return the image-values for the given input-coordinates."""
        # Use standard coordinate tensor if none is given
        coordinate_tensor = self.makeCoordinateTensor(output_shape)
    
        if moving_image is None:
            moving_image = self.moving_image
        
        # Initialize images to transform
        transformed_image = torch.zeros(coordinate_tensor.shape[0]).cuda()
        if moving_mask is not None:
            moving_mask = torch.FloatTensor(moving_mask.astype(float)).cuda()
            transformed_mask = torch.zeros(coordinate_tensor.shape[0]).cuda()

        with torch.no_grad():
            index = 0
            for grid_batch in torch.split(coordinate_tensor, forward_batch_size):
                output_batch = self.network(grid_batch)
                coord_temp = torch.add(output_batch, grid_batch)
                output_batch = coord_temp   
                transformed_image[index:index + forward_batch_size] = self.interpolate(moving_image, output_batch, method = 'trilinear')
                if moving_mask is not None:
                    transformed_mask[index:index + forward_batch_size] = self.interpolate(moving_mask, output_batch, method = 'nearest')
                index = index + forward_batch_size
            transformed_image = transformed_image.reshape(output_shape)
            if moving_mask is not None:
                transformed_mask = transformed_mask.reshape(output_shape)
                
        if moving_mask is not None:
            return transformed_image.cpu().detach().numpy(), transformed_mask.cpu().detach().numpy().astype(np.uint8)
        else:
            return transformed_image.cpu().detach().numpy()

    def __init__(self, moving_image, fixed_image, **kwargs):
        """Initialize the learning model."""
        # Set all default arguments in a dict: self.args
        self.setDefaultArguments()

        # Check if all kwargs keys are valid (this checks for typos)
        assert all(kwarg in self.args.keys() for kwarg in kwargs)

        # Parse important argument from kwargs
        self.log_interval = kwargs['log_interval'] if 'log_interval' in kwargs else self.args['log_interval']
        self.verbose = kwargs['verbose'] if 'verbose' in kwargs else self.args['verbose'] 
        self.save_folder = kwargs['save_folder'] if 'save_folder' in kwargs else self.args['save_folder']
        # Make folder for output
        if not self.save_folder ==  '' and not os.path.isdir(self.save_folder):
            os.mkdir(self.save_folder)
        self.gpu = kwargs['gpu'] if 'gpu' in kwargs else self.args['gpu']

        self.loss_function_arg = kwargs['loss_function'] if 'loss_function' in kwargs else self.args['loss_function']

        self.mask = kwargs['mask'] if 'mask' in kwargs else self.args['mask']

        self.jacobian_regularization = kwargs['jacobian_regularization'] if 'jacobian_regularization' in kwargs else self.args['jacobian_regularization']
        self.alpha_jacobian = kwargs['alpha_jacobian'] if 'alpha_jacobian' in kwargs else self.args['alpha_jacobian']
        self.reg_norm_jacobian = kwargs['reg_norm_jacobian'] if 'reg_norm_jacobian' in kwargs else self.args['reg_norm_jacobian']

        self.optimizer_arg = kwargs['optimizer'] if 'optimizer' in kwargs else self.args['optimizer']
        self.epochs = kwargs['epochs'] if 'epochs' in kwargs else self.args['epochs']
        self.lr = kwargs['lr'] if 'lr' in kwargs else self.args['lr']
        self.momentum = kwargs['momentum'] if 'momentum' in kwargs else self.args['momentum']
        
        self.layers = kwargs['layers'] if 'layers' in kwargs else self.args['layers']
        self.omega = kwargs['omega'] if 'omega' in kwargs else self.args['omega']
        
        self.offset = kwargs['offset'] if 'offset' in kwargs else self.args['offset']

        # Add slash to divide folder and filename
        self.save_folder +=  os.path.sep
        self.save_folder +=  os.path.sep

        # Make loss list to save losses
        self.loss_list = [0 for _ in range(self.epochs)]
        self.data_loss_list = [0 for _ in range(self.epochs)]
        
        # Set seed
        torch.manual_seed(self.args['seed'])

        # Load network
        self.network_from_file = kwargs['network'] if 'network' in kwargs else self.args['network']
        if self.network_from_file is None:
            self.network = networks.Siren(self.layers, self.omega)
            # self.network = networks.MLP(self.layers)   
            
        else:
            self.network = torch.load(self.network_from_file)
            if self.gpu:
                self.network.cuda()

        # Choose the optimizer
        if self.optimizer_arg.lower() ==  'sgd':
            self.optimizer = optim.SGD(self.network.parameters(), lr = self.lr, momentum = self.momentum)

        elif self.optimizer_arg.lower() ==  'adam':
            self.optimizer = optim.Adam(self.network.parameters(), lr = self.lr)

        elif self.optimizer_arg.lower() ==  'adadelta':
            self.optimizer = optim.Adadelta(self.network.parameters(), lr = self.lr)

        elif self.optimizer_arg.lower() ==  'rmsprop':
            self.optimizer = optim.RMSprop(self.network.parameters(), lr = self.lr)            
            
        else:
            self.optimizer = optim.SGD(self.network.parameters(), lr = self.lr, momentum = self.momentum)
            print('WARNING: ' + str(self.optimizer_arg) + ' not recognized as optimizer, picked SGD instead')

        
        # Choose the loss function
        if self.loss_function_arg.lower() ==  'mse':
            self.criterion = nn.MSELoss()

        elif self.loss_function_arg.lower() ==  'l1':
            self.criterion = nn.L1Loss()

        elif self.loss_function_arg.lower() ==  'ncc':
            self.criterion = ncc.NCC()            
            
        elif self.loss_function_arg.lower() ==  'smoothl1':
            self.criterion = nn.SmoothL1Loss(beta = 0.1)
            
        elif self.loss_function_arg.lower() ==  'huber':
            self.criterion = nn.HuberLoss()            

        else:
            self.criterion = nn.MSELoss()
            print('WARNING: ' + str(self.loss_function_arg) + ' not recognized as loss function, picked MSE instead')

        # Move variables to GPU
        if self.gpu:
            self.network.cuda()
            
        self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else self.args['batch_size']
        
        self.moving_image = moving_image
        self.fixed_image = fixed_image
        if self.gpu:
            self.moving_image = self.moving_image.cuda()
            self.fixed_image = self.fixed_image.cuda()
            
        self.image_shape = (int(self.fixed_image.shape[0]), int(self.fixed_image.shape[1]), int(self.fixed_image.shape[2]))
        self.possible_coordinate_tensor = self.makeCoordinateTensor(self.fixed_image.shape, self.gpu, self.mask)
        
        self.ncc_width = kwargs['ncc_width'] if 'ncc_width' in kwargs else self.args['ncc_width']
        coordinate_tensor_loc = [torch.linspace(-self.ncc_width, self.ncc_width, 5) for i in range(3)]
        coordinate_tensor_loc = torch.meshgrid(*coordinate_tensor_loc)
        coordinate_tensor_loc = torch.stack(coordinate_tensor_loc, dim = 3)
        coordinate_tensor_loc = coordinate_tensor_loc.view([5**3, 3])
        coordinate_tensor_loc = torch.tile(coordinate_tensor_loc, (int(self.batch_size/(5**3)), 1))
        
        if self.gpu:
            self.coordinate_tensor_loc = coordinate_tensor_loc.cuda()       
            
        if self.verbose:
            image_shape = (int(self.fixed_image.shape[0]), int(self.fixed_image.shape[1]), int(self.fixed_image.shape[2]))
            self.image_shape_verbose = tuple(map(lambda item: int(item / 1), image_shape))
            self.coordinate_tensor = self.makeCoordinateTensor(self.image_shape_verbose)
            self.fixed_image_tensor = self.interpolate(self.fixed_image, self.coordinate_tensor, method = 'trilinear')
            self.fixed_image_tensor = self.fixed_image_tensor.detach().cpu().view(self.image_shape_verbose) 
            
        if self.gpu:
            self.offset = torch.FloatTensor(self.offset).cuda()
            
    def setDefaultArguments(self):
        """Set default arguments."""

        self.args = {}

        self.args['network'] = None
        self.args['epochs'] = 1000
        
        self.args['log_interval'] = 250
        self.args['verbose'] = False
        self.args['save_folder'] = 'output'
        self.args['gpu'] = torch.cuda.is_available()
        
        self.args['mask'] = None

        self.args['loss_function'] = 'ncc'
        self.args['ncc_width'] = 0.025

        self.args['jacobian_regularization'] = True
        self.args['alpha_jacobian'] = 0.05
        self.args['reg_norm_jacobian'] = 1
        
        self.args["hyper_regularization"] = False
        self.args["alpha_hyper"] = 0.25

        self.args["bending_regularization"] = False
        self.args["alpha_bending"] = 10.0


        self.args['optimizer'] = 'Adam'
        self.args['lr'] = 1e-5
        self.args['momentum'] = 0.5
        
        self.args['batch_size'] = 100*(5**3)

        self.args['layers'] = [3, 32, 32, 32, 3]
        self.args['omega'] = 30
        
        self.args['seed'] = 1
        self.args['offset'] = [0, 0, 0]
        
    def cuda(self):
        """Move the model to the GPU."""

        self.gpu = True
        self.network.cuda()
        
    def gradient(self, input_coords, output, grad_outputs = None):
        """Compute the gradient of the output wrt the input."""

        grad_outputs = torch.ones_like(output)
        grad = torch.autograd.grad(output, [input_coords], grad_outputs = grad_outputs, create_graph = True)[0]
        return grad
        
    def makeCoordinateTensor(self, dims = (28, 28, 28), gpu = True, mask = None):
        """Make a coordinate tensor."""

        coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
        coordinate_tensor = torch.meshgrid(*coordinate_tensor)
        coordinate_tensor = torch.stack(coordinate_tensor, dim = 3)
        coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])
        
        if mask is not None:
            coordinate_tensor = coordinate_tensor[mask.flatten() > 0, :]      

        # Move to GPU if necessary
        if self.gpu and gpu:
            coordinate_tensor = coordinate_tensor.cuda()
        return coordinate_tensor
        
    def interpolate(self, input_array, coordinates, method = 'trilinear'):
        return interpolation.fast_interpolation(input_array, coordinates[:, 0], coordinates[:, 1], coordinates[:,2], method)

    def train(self, epochs = None):
        """Train the network."""
        # Determine epochs
        if epochs is None:
            epochs = self.epochs

        # Set seed
        torch.manual_seed(self.args['seed'])

        # Extend lost_list if necessary
        if not len(self.loss_list) ==  epochs:
            self.loss_list = [0 for _ in range(epochs)]
            self.data_loss_list = [0 for _ in range(epochs)]
        
        # Perform training iterations
        for i in tqdm.tqdm(range(epochs)):
            self.trainingIteration(i)
            
    def trainingIteration(self, epoch):
        """Perform one iteration of training."""
        update_flag = 0
        # Reset the gradient
        self.network.train()

        loss = 0
        indices = torch.randperm(self.possible_coordinate_tensor.shape[0], device = 'cuda')[:int(self.batch_size/(5**3))]
        coordinate_tensor = self.possible_coordinate_tensor[indices, :]
        coordinate_tensor = torch.repeat_interleave(coordinate_tensor, torch.tensor(np.ones(int(self.batch_size/(5**3)))*(5**3)).int().cuda(), dim = 0)    
        coordinate_tensor = coordinate_tensor + self.coordinate_tensor_loc   
        coordinate_tensor = coordinate_tensor.requires_grad_(True) 

        output = self.network(coordinate_tensor) + self.offset
        coord_temp = torch.add(output, coordinate_tensor)
        output = coord_temp
        
        if epoch ==  0:
            if self.verbose:
                with torch.no_grad():
                    fig = visualizing.plot_aligned_mips(self.moving_image,self.fixed_image)
                    fig.tight_layout()
                    fig.savefig(r'{}\overlay_{}.png'.format(self.save_folder,str(0)))
                    fig.savefig(r'{}\overlay_{}.svg'.format(self.save_folder,str(0)))
                    plt.close()

        transformed_image = self.interpolate(self.moving_image, coord_temp, method = 'trilinear')
        fixed_image = self.interpolate(self.fixed_image, coordinate_tensor, method = 'trilinear')

        # Compute the loss
        loss +=  self.criterion(transformed_image, fixed_image)
        
        # Store the value of the data loss
        self.data_loss_list[epoch] = loss.detach().cpu().numpy()
                
        # Relativation of output
        output_rel = torch.subtract(output, coordinate_tensor)       

        # Regularization only central points coordinates
        if self.jacobian_regularization: 
            loss +=  self.alpha_jacobian * regularizers.compute_jacobian_loss(coordinate_tensor, output_rel, self.batch_size)
            
        # Perform the backpropagation and update the parameters accordingly
        # self.optimizer.zero_grad()
        for param in self.network.parameters():
            param.grad = None        
        loss.backward()
        
        self.optimizer.step()

        # Store the value of the total loss
        self.loss_list[epoch] = loss.detach().cpu().numpy()

        # Print Logs
        if (epoch % self.log_interval ==  0 or epoch ==  self.epochs - 1 or update_flag ==  1):
            if self.verbose:
                with torch.no_grad():  
                    # Apply INR transformation to the moving image (using batches of coordinates to avoid out of GPU memory)
                    transformed_image = torch.zeros(self.coordinate_tensor.shape[0]).cuda()

                    with torch.no_grad():
                        forward_batch_size = 30000;
                        index = 0
                        for grid_batch in torch.split(self.coordinate_tensor, forward_batch_size):
                            output_batch = self.network(grid_batch)
                            coord_temp = torch.add(output_batch, grid_batch)
                            output_batch = coord_temp   
                            transformed_image[index:index + forward_batch_size] = self.interpolate(self.moving_image, output_batch, method = 'trilinear')
                            index = index + forward_batch_size
                    self.printLogs(epoch, loss, transformed_image, output, coordinate_tensor.detach().cpu().numpy())
                    
    def printLogs(self, epoch, loss, transformed_image, output, locations):
        """Print the progress of the training."""

        # Print Loss
        print("-" * 10 + "  epoch: " + str(epoch+1) + "  " + "-" * 10)
        print("Loss: " + str(loss.detach().cpu().numpy()))
            
        # Reshape transformed image
        moved_image_plot = transformed_image.cpu().detach().numpy().reshape(*self.image_shape_verbose)
        fixed_image_plot = self.fixed_image.detach().cpu().numpy()
        
        fig = visualizing.plot_aligned_mips(moved_image_plot,fixed_image_plot)
        fig.tight_layout()
        fig.savefig(r'{}\overlay_{}.png'.format(self.save_folder,str(epoch+1)))
        fig.savefig(r'{}\overlay_{}.svg'.format(self.save_folder,str(epoch+1)))
        plt.close()
        
        x_locations = locations[:, 0]
        y_locations = locations[:, 1]
        z_locations = locations[:, 2]
        x_locations = (x_locations + 1) * (self.image_shape[0]-1) * 0.5
        y_locations = (y_locations + 1) * (self.image_shape[1]-1) * 0.5
        z_locations = (z_locations + 1) * (self.image_shape[2]-1) * 0.5
        
        plt.figure()
        plt.imshow(np.max(fixed_image_plot,axis = 2), cmap = 'gray') 
        plt.scatter(y_locations, x_locations, c = 'b', s = 1, marker = 'o')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(r'{}\cor_fixed_{}.png'.format(self.save_folder, str(epoch+1).zfill(5)), dpi = 300, bbox_inches = 'tight')        
        plt.close()
        
        plt.figure()
        plt.imshow(np.max(fixed_image_plot,axis = 0), cmap = 'gray') 
        plt.scatter(z_locations, x_locations, c = 'b', s = 1, marker = 'o')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(r'{}\axi_fixed_{}.png'.format(self.save_folder, str(epoch+1).zfill(5)), dpi = 300, bbox_inches = 'tight')        
        plt.close()