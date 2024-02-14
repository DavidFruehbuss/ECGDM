import torch
import numpy as np

import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean

"""
This file implements the generative framework [diffusion model] for the model

Framework:

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

Data:

data_samples: molecule, protein_pocket (graph batched)

batch_graph_size_mol = sum([num_nodes for graph in batch])
batch_graph_size_pro = sum([num_nodes for graph in batch])

before concatination:
molecule: {'x': 3-D position, 'h': one_hot atom_types, 
            'idx': node to graph mapping, size: molecule sizes} 
            (first_dim x, h = batch_graph_size_mol)
protein_pocket: {'x': 3-D position, 'h': one_hot residue_types, 
            'idx': node to graph mapping, protein_pocket: molecule sizes} 
            (first_dim x, h = batch_graph_size_pro)

after concatination:
    molecule: [batch_graph_size_mol,  n_dim = 3 + num_atoms]
    protein_pocket: [batch_graph_size_pro, n_dim = 3 + num_residues]



"""

class Conditional_Diffusion_Model(nn.Modules):

    """
    Conditional Denoising Diffusion Model

    # need to handle case where t = 0
    
    """

    def __init__(
        self,
        neural_net: nn.Module,
        timesteps: int,
        num_atoms: int,
        num_residues: int,
    ):
        """
        Parameters:

        
        """

        super().__init__()
        
        self.neural_net = neural_net()
        self.T = timesteps

        # dataset info
        self.num_atoms = num_atoms
        self.num_residues = num_residues
        self.x_dim = 3

        
    def forward(self, z_data):

        molecule, protein_pocket = z_data

        # compute noised sample
        z_t_mol, z_t_pro, epsilon_mol, epsilon_pro, t = self.noise_process(z_data)

        # use neural network to predict noise
        epsilon_hat_mol, epsilon_hat_pro = self.neural_net(z_t_mol, z_t_pro, t, molecule['idx'], protein_pocket['idx'])

        # compute denoised sample
        # z_data_hat = (1 / alpha_t) * z_t - (sigma_t / alpha_t) * epsilon_hat

        # compute the sum squared error loss per graph
        error_mol = scatter_add(torch.sum((epsilon_mol - epsilon_hat_mol)**2, dim=-1), molecule['idx'], dim=0)
        error_pro = scatter_add(torch.sum((epsilon_pro - epsilon_hat_pro)**2, dim=-1)**2, protein_pocket['idx'], dim=0)

        # normalize the graph_loss by graph size
        error_mol = error_mol / ((molecule['x'].size(0) + self.num_atoms) * molecule['size'])
        error_pro = error_pro / ((protein_pocket['x'].size(0) + self.num_residues * protein_pocket['size']))

        loss_t = 0.5 * (error_mol + error_pro)

        return loss_t

    def noise_process(self, z_data):

        """
        Creates noised samples from data_samples following a predefined noise schedule
        """

        molecule, protein_pocket = z_data
        batch_size = molecule['size'].size(0)
        device = z_data.device

        # TODO: add normalisation (not sure why yet, so leave it for later)

        # sample t ~ U(0,...,T) for each graph individually
        t = torch.randint(0, self.T + 1, size=batch_size, device=device) # is an int
        s = t - 1

        # noise schedule
        alpha_t = 1 - (t / self.T)**2
        sigma_t = torch.sqrt(1 - alpha_t**2)

        # prepare joint point cloud
        xh_mol = torch.cat(molecule['x'], molecule['h'], dim=1)
        xh_pro = torch.cat(protein_pocket['x'], protein_pocket['h'], dim=1)

        # compute noised sample z_t
        # for x cord. we mean center the normal noise for each graph
        x_noise = torch.randn(size=(len(xh_mol) + len(xh_pro), self.x_dim), device=device)
        eps_x = x_noise - scatter_mean(x_noise, torch.cat((molecule['idx'], protein_pocket['idx'])), dim=0)
        # for h we need standard normal noise
        eps_h_mol = torch.randn(size=(len(xh_mol), self.num_atoms), device=device)
        eps_h_pro = torch.randn(size=(len(xh_pro), self.num_residues), device=device)
        epsilon_mol = torch.cat([eps_x[:len(xh_mol)], eps_h_mol], dim=1)
        epsilon_pro = torch.cat([eps_x[len(xh_mol):], eps_h_pro], dim=1)

        # compute noised representations
        z_t_mol = alpha_t[molecule['idx']] * xh_mol - sigma_t[molecule['idx']] * epsilon_mol
        z_t_pro = alpha_t[protein_pocket['idx']] * xh_pro - sigma_t[protein_pocket['idx']] * epsilon_pro

        return z_t_mol, z_t_pro, epsilon_mol, epsilon_pro, t