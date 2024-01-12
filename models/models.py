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

License: MIT
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

ScaleIntensity = monai.transforms.ScaleIntensity(minv = 0.0, maxv = 1.0)

class ImplicitRegistrator:

    def __call__(self, output_shape = (28, 28, 28), moving_image = None, moving_mask = None, forward_batch_size = 10000):

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

        self.frangi_sigmas = kwargs['frangi_sigmas'] if 'frangi_sigmas' in kwargs else self.args['frangi_sigmas']
        self.frangi_epochs = kwargs['frangi_epochs'] if 'frangi_epochs' in kwargs else self.args['frangi_epochs']
        self.frangi_options = kwargs['frangi_options'] if 'frangi_options' in kwargs else self.args['frangi_options']
        self.aim_options = kwargs['aim_options'] if 'aim_options' in kwargs else self.args['aim_options']
        
        self.loss_function_arg = kwargs['loss_function'] if 'loss_function' in kwargs else self.args['loss_function']

        self.mask = kwargs['mask'] if 'mask' in kwargs else self.args['mask']

        # Parse regularization kwargs
        self.jacobian_regularization = (
            kwargs["jacobian_regularization"]
            if "jacobian_regularization" in kwargs
            else self.args["jacobian_regularization"]
        )
        self.alpha_jacobian = (
            kwargs["alpha_jacobian"]
            if "alpha_jacobian" in kwargs
            else self.args["alpha_jacobian"]
        )

        self.reg_norm_jacobian = (
            kwargs["reg_norm_jacobian"]
            if "reg_norm_jacobian" in kwargs
            else self.args["reg_norm_jacobian"]
        )
        
        self.hyper_regularization = (
            kwargs["hyper_regularization"]
            if "hyper_regularization" in kwargs
            else self.args["hyper_regularization"]
        )
        self.alpha_hyper = (
            kwargs["alpha_hyper"]
            if "alpha_hyper" in kwargs
            else self.args["alpha_hyper"]
        )

        self.bending_regularization = (
            kwargs["bending_regularization"]
            if "bending_regularization" in kwargs
            else self.args["bending_regularization"]
        )
        self.alpha_bending = (
            kwargs["alpha_bending"]
            if "alpha_bending" in kwargs
            else self.args["alpha_bending"]
        )

        self.optimizer_arg = kwargs['optimizer'] if 'optimizer' in kwargs else self.args['optimizer']
        self.epochs = kwargs['epochs'] if 'epochs' in kwargs else self.args['epochs']
        self.lr = kwargs['lr'] if 'lr' in kwargs else self.args['lr']
        self.momentum = kwargs['momentum'] if 'momentum' in kwargs else self.args['momentum']
        self.scheduler_ms = kwargs['scheduler_ms'] if 'scheduler_ms' in kwargs else self.args['scheduler_ms']
        self.scheduler_gamma = kwargs['scheduler_gamma'] if 'scheduler_gamma' in kwargs else self.args['scheduler_gamma']
        
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

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.scheduler_ms, self.scheduler_gamma)
        
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
        
        self.ncc_widths = kwargs['ncc_widths'] if 'ncc_widths' in kwargs else self.args['ncc_widths']      
            
        if self.verbose:
            image_shape = (int(self.fixed_image.shape[0]), int(self.fixed_image.shape[1]), int(self.fixed_image.shape[2]))
            self.image_shape_verbose = tuple(map(lambda item: int(item / 1), image_shape))
            self.coordinate_tensor = self.makeCoordinateTensor(self.image_shape_verbose)
            self.fixed_image_tensor = self.interpolate(self.fixed_image, self.coordinate_tensor, method = 'trilinear')
            self.fixed_image_tensor = self.fixed_image_tensor.detach().cpu().view(self.image_shape_verbose) 
            
        if self.gpu:
            self.offset = torch.FloatTensor(self.offset).cuda()
            
    def setDefaultArguments(self):

        self.args = {}

        self.args['network'] = None
        self.args['epochs'] = 1000
        
        self.args['log_interval'] = 250
        self.args['verbose'] = False
        self.args['save_folder'] = 'output'
        self.args['gpu'] = torch.cuda.is_available()
        
        self.args['mask'] = None

        self.args['frangi_sigmas'] = np.array([9, 7, 5, 3])
        frangi_interval = np.array([0.25, 0.25, 0.25, 0.25])
        self.args['frangi_epochs'] = np.insert(np.cumsum(np.floor(frangi_interval*self.args['epochs']))[:-1],0,0)
        frangi_options = dict()
        frangi_options['alpha'] = 0.5
        frangi_options['beta'] = 0.5
        frangi_options['gamma'] = 0.05
        frangi_options['bw_flag'] = True # White voxels are vessels
        self.args['frangi_options'] = frangi_options
        
        aim_options = dict()
        aim_options['half_size_win'] = 5
        aim_options['min_sd'] = 0.2
        self.args['aim_options'] = aim_options

        self.args['loss_function'] = 'ncc'
        self.args['ncc_widths'] = [0.75, 0.5, 0.25, 0.125]

        self.args['jacobian_regularization'] = False
        self.args['alpha_jacobian'] = 0.05
        self.args['reg_norm_jacobian'] = 1
        
        self.args["hyper_regularization"] = False
        self.args["alpha_hyper"] = 0.25

        self.args["bending_regularization"] = False
        self.args["alpha_bending"] = 10.0

        self.args['optimizer'] = 'Adam'
        self.args['lr'] = 1e-5
        self.args['momentum'] = 0.5
        self.args['scheduler_ms'] = self.args['frangi_epochs']
        self.args['scheduler_gamma'] = 0.9
        
        self.args['batch_size'] = 100*(5**3)

        self.args['layers'] = [3, 32, 32, 32, 3]
        self.args['omega'] = 30
        
        self.args['seed'] = 1
        self.args['offset'] = [0, 0, 0]
        
    def cuda(self):

        self.gpu = True
        self.network.cuda()
        
    def gradient(self, input_coords, output, grad_outputs = None):

        grad_outputs = torch.ones_like(output)
        grad = torch.autograd.grad(output, [input_coords], grad_outputs = grad_outputs, create_graph = True)[0]
        return grad
        
    def makeCoordinateTensor(self, dims = (28, 28, 28), gpu = True, mask = None):

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
        
    def update_images(self):

        with torch.no_grad():
            self.frangi_options['sigma'] = self.frangi_sigmas[self.frangi_iter]
            self.curr_moving_image = ScaleIntensity(processing.frangi_aim(self.moving_image, self.frangi_options, self.aim_options, gpu = 'cuda'))
            self.curr_fixed_image = ScaleIntensity(processing.frangi_aim(self.fixed_image, self.frangi_options, self.aim_options, gpu = 'cuda'))
            coordinate_tensor_loc = [torch.linspace(-self.ncc_widths[self.frangi_iter], self.ncc_widths[self.frangi_iter], 5) for i in range(3)]
            coordinate_tensor_loc = torch.meshgrid(*coordinate_tensor_loc)
            coordinate_tensor_loc = torch.stack(coordinate_tensor_loc, dim = 3)
            coordinate_tensor_loc = coordinate_tensor_loc.view([5**3, 3])
            coordinate_tensor_loc = torch.tile(coordinate_tensor_loc, (int(self.batch_size/(5**3)), 1))
            if self.gpu:
                self.coordinate_tensor_loc = coordinate_tensor_loc.cuda() 
                
    def train(self, epochs = None):

        # Determine epochs
        if epochs is None:
            epochs = self.epochs

        # Set seed
        torch.manual_seed(self.args['seed'])

        # Extend lost_list if necessary
        if not len(self.loss_list) ==  epochs:
            self.loss_list = [0 for _ in range(epochs)]
            self.data_loss_list = [0 for _ in range(epochs)]
        
        self.frangi_iter = 0
        # Perform training iterations
        for i in tqdm.tqdm(range(epochs)):
            self.trainingIteration(i)
            
    def trainingIteration(self, epoch):

        update_flag = 0
        # Reset the gradient
        self.network.train()
        
        # Update images with current Frangi sigma
        if epoch in self.frangi_epochs:
            self.update_images()
            self.frangi_iter = self.frangi_iter + 1
            update_flag = 1
            
        if epoch in self.frangi_epochs-1:
            update_flag = 1 
            
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
                    fig = visualizing.plot_aligned_mips(self.curr_moving_image,self.curr_fixed_image)
                    fig.tight_layout()
                    fig.savefig(r'{}\overlay_{}.png'.format(self.save_folder,str(0)))
                    fig.savefig(r'{}\overlay_{}.svg'.format(self.save_folder,str(0)))
                    plt.close()

        transformed_image = self.interpolate(self.curr_moving_image, coord_temp, method = 'trilinear')
        fixed_image = self.interpolate(self.curr_fixed_image, coordinate_tensor, method = 'trilinear')

        # Compute the loss
        loss +=  self.criterion(transformed_image, fixed_image)
        
        # Store the value of the data loss
        self.data_loss_list[epoch] = loss.detach().cpu().numpy()
                
        # Relativation of output
        output_rel = torch.subtract(output, coordinate_tensor)       

        # Regularization
        if self.jacobian_regularization:
            loss += self.alpha_jacobian * regularizers.compute_jacobian_loss(
                coordinate_tensor, output_rel, batch_size=self.batch_size
            )
        if self.hyper_regularization:
            loss += self.alpha_hyper * regularizers.compute_hyper_elastic_loss(
                coordinate_tensor, output_rel, batch_size=self.batch_size
            )
        if self.bending_regularization:
            loss += self.alpha_bending * regularizers.compute_bending_energy(
                coordinate_tensor, output_rel, batch_ssize=self.batch_size
            )
            
        # Perform the backpropagation and update the parameters accordingly
        self.optimizer.zero_grad()
        for param in self.network.parameters():
            param.grad = None        
        loss.backward()
        
        # Optimizer and scheduler step
        self.optimizer.step()
        self.scheduler.step()

        # Store the value of the total loss
        self.loss_list[epoch] = loss.detach().cpu().numpy()

        # Print Logs
        if (epoch % self.log_interval ==  0 or epoch ==  self.epochs - 1 or update_flag ==  1):
            if self.verbose:
                with torch.no_grad():  
                
                    # Apply INR transformation to the moving image (using batches of coordinates to avoid out of GPU memory)
                    transformed_image = torch.zeros(self.coordinate_tensor.shape[0]).cuda()
                    scale_of_axes = [(0.5 * s) for s in self.image_shape_verbose]
                    displacement_field = torch.zeros_like(self.coordinate_tensor)
                    
                    with torch.no_grad():
                        forward_batch_size = 30000;
                        index = 0
                        for grid_batch in torch.split(self.coordinate_tensor, forward_batch_size):
                            output_batch = self.network(grid_batch)
                            coord_temp = torch.add(output_batch, grid_batch)
                            output_batch = coord_temp   
                            transformed_image[index:index + forward_batch_size] = self.interpolate(self.curr_moving_image, output_batch, method = 'trilinear')
                            displacement_field[index:index + forward_batch_size,:] = self.network(grid_batch)
                            index = index + forward_batch_size
                    displacement_field = displacement_field.cpu().detach().numpy() * (scale_of_axes)
                    np.save(r'{}\df_{}.npy'.format(self.save_folder, str(epoch+1)), np.float16(displacement_field)) 
                    self.printLogs(epoch, loss, transformed_image, output, coordinate_tensor.detach().cpu().numpy())
                    
    def printLogs(self, epoch, loss, transformed_image, output, locations):
        """Print the progress of the training."""

        # Print Loss
        print("-" * 10 + "  epoch: " + str(epoch+1) + "  " + "-" * 10)
        print("Loss: " + str(loss.detach().cpu().numpy()))
            
        # Reshape transformed image
        moved_image_plot = transformed_image.cpu().detach().numpy().reshape(*self.image_shape_verbose)
        fixed_image_plot = self.curr_fixed_image.detach().cpu().numpy()
        
        # Plot MIPs
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
        
        # Plot sampled points
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