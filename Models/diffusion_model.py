import torch
import numpy as np

import torch.nn as nn
inport torch.nn.functional as F

"""
This file implements the generative framework [diffusion model] for the model

Given Z_data we create noised samples z_t with t ~ U(0,...,T).
We use the noise process q(z_t|z_data) = N(z_t|alpha_t * z_data, sigma_t**2 * I).
This gives us z_t = alpha_t * z_t - (sigma_t / alpha_t) * epsilon with epsilon ~ N(0|I).

We use z_t as input to our denoising neural network (NN) to predict epsilon.
-> NN(z_t, t) = epsilon_hat
The loss of the neural network is simply L_train = ||epsilon - epsilon_hat||**2.

(During training we train on the model on one step transitions t -> s, s = t-1 to 
get a stepwise denoising process. )

Any one step transition is given by:
q(z_s|z_t, z_data) = N( z_s | (alpha_t_s * sigma_s**2 / sigma_t**2) * z_t 
    + (alpha_s * sigma_t_s**2 / sigma_t**2) * z_data, (sigma_t_s**2 * sigma_t_s**2 / sigma_t**2) * I )
with alpha_t_s = alpha_t / alpha_s and sigma_t_s**2 = sigma_t**2 - alpha_t_s**2 * sigma_s**2.

"""

class Conditional_Diffusion_Model(nn.Modules):

    """
    Conditional Denoising Diffusion Model
    """

    def __init__(
        self,
        neural_net: nn.Module,
        timesteps: int,
            
    ):
        self.neural_net = neural_net()
        self.T = timesteps

        # TODO: Define num_categories for molecules and proteins
        # self.num_atoms = 
        # self.num_residues = 
        self.x_dim = 3

        
    def forward(self, z_data):

        # compute noised sample
        z_t_mol, z_t_pro, epsilon_mol, epsilon_pro, alpha_t, sigma_t, t = self.noise_process(z_data)

        # use neural network to predict noise
        epsilon_hat_mol, epsilon_hat_pro = self.neural_net(z_t_mol, z_t_pro)

        # compute denoised sample
        # z_data_hat = (1 / alpha_t) * z_t - (sigma_t / alpha_t) * epsilon_hat

        # TODO: need to check what I want to do here
        epsilon = [epsilon_mol, epsilon_pro]
        epsilon_hat = [epsilon_hat_mol, epsilon_hat_pro]

        loss = self.loss_fn(epsilon_hat, epsilon)

        return loss

    def noise_process(self, z_data):

        """
        Creates noised samples from data_samples

        data_samples: molecule, protein_pocket (graph batched)

        batch_graph_size_mol = sum([num_nodes for graph in batch])
        batch_graph_size_pro = sum([num_nodes for graph in batch])

        before concatination:
        molecule: {'x': 3-D position, 'h': one_hot atom_types, 
                    'mask': node to graph mapping, size: molecule sizes} 
                    (first_dim x, h = batch_graph_size_mol)
        protein_pocket: {'x': 3-D position, 'h': one_hot residue_types, 
                    'mask': node to graph mapping, protein_pocket: molecule sizes} 
                    (first_dim x, h = batch_graph_size_pro)

        after concatination:
            molecule: [batch_graph_size_mol, num_nodes = 9, n_dim = 3 + num_atoms]
            protein_pocket: [batch_graph_size_pro, num_nodes = ?, n_dim = 3 + num_residues]

        # currently for batch_size = 1
        # currently no device handling
        """

        molecule, protein_pocket = z_data
        batch_size = molecule['size'].size(0)

        # TODO: add normalisation

        # sample t ~ U(0,...,T)
        t = torch.randint(0, self.T + 1, size=batch_size) # is an int
        s = t - 1

        # noise schedule
        alpha_t = 1 - (t / self.T)**2
        sigma_t = torch.sqrt(1 - alpha_t**2)

        # prepare joint point cloud
        xh_mol = torch.cat(molecule['x'], molecule['h'], dim=1)
        xh_pro = torch.cat(protein_pocket['x'], protein_pocket['h'], dim=1)

        # compute noised sample z_t
        # for x cord. we need mean_cetered normal noise
        eps_x_mol = torch.randn(size=(len(xh_mol) + len(xh_pro), self.x_dim))
        # TODO: joint mean_centering? see remove mean batch
        # for h we need standard normal noise
        eps_h_mol = torch.randn(size=(len(xh_mol), self.num_atoms))
        eps_h_pro = torch.randn(size=(len(xh_pro), self.num_residues))
        epsilon_mol = torch.cat([eps_x_mol[:len(xh_mol)], eps_h_mol], dim=1)
        epsilon_pro = torch.cat([eps_x_mol[len(xh_mol):], eps_h_pro], dim=1)

        # TODO: figure out why it makes sense to put indexing on alpha, sigma
        z_t_mol = alpha_t[molecule['mask']] * xh_mol - sigma_t * epsilon_mol
        z_t_pro = alpha_t[protein_pocket['mask']] * xh_pro - sigma_t * epsilon_pro

        return z_t_mol, z_t_pro, epsilon_mol, epsilon_pro, alpha_t, sigma_t, t
    
    def loss_fn(epsilon_hat, epsilon):
        # need to modify for point cloud data
        loss = (epsilon_hat - epsilon)**2
        return loss