import torch
import numpy

import torch.nn as nn
inport torch.nn.functional as F

"""
This file implements the generative framework [diffusion model] for the model

Given Z_data we create noised samples z_t with t ~ U(0,...,T).
We use the noise process q(z_t|z_data) = N(z_t|alpha_t * z_data, sigma_t**2 * I).
This gives us z_t = alpha_t * z_t - (sigma_t / alpha_t) * epsilon with epsilon ~ N(0|I).

We use z_t as input to our denoising neural network (NN). Additionally we give the time_step t.
-> NN(z_t, t) = epsilon_hat
The loss of the neural network is simply L_train = ||epsilon - epsilon_hat||**2.
"""

class Diffusion_Model(nn.Modules):

    """
    Denoising Diffusion Model
    """

    def __init__(
        self,
        neural_net,
        T,
            
    ):
        self.neural_net = neural_net
        self.T = T
        
    def forward(self, z_data):

        # compute noised sample
        z_t, t = self.noise_process(z_data)

        # compute alpha_t and sigma_t again
        alpha_t = 1 - (t / self.T)**2 # additional tricks for numerical stability needed
        sigma_t = torch.sqrt(1 - alpha_t**2)

        # use neural network to predict noise
        epsilon_hat = self.neural_net(z_t)

        # compute denoised sample
        z_data_hat = (1 / alpha_t) * z_t - (sigma_t / alpha_t) * epsilon_hat

        return z_data_hat

    def noise_process(self, z_data):

        """
        Creates noised samples from data samples

        # currently for batch_size = 1
        """

        # sample t ~ U(0,...,T)
        t = torch.rand() * self.T

        # noise schedule
        alpha_t = 1 - (t / self.T)**2 # additional tricks for numerical stability needed
        sigma_t = torch.sqrt(1 - alpha_t**2)

        # compute noised sample z_t
        epsilon = torch.normal(0,1)
        z_t = alpha_t * z_data - sigma_t * epsilon
        return z_t, t
    
    def loss_fn(z_t, z_data):
        loss = (z_t - z_data)**2 # need to modify for point cloud data
        return loss