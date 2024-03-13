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

Loss functions:

first simple loss: essentially just loss_t, which is for training only if features are fixed (KL=0)

l2 loss: loss_0, loss_t and KL combined

vlb loss: full vlb loss (adds 4 additional loss terms and no normalisation) 

"""

class Conditional_Diffusion_Model(nn.Module):

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
        norm_values: list
    ):
        """
        Parameters:

        
        """

        super().__init__()
        
        self.neural_net = neural_net
        self.T = timesteps

        # dataset info
        self.num_atoms = num_atoms
        self.num_residues = num_residues
        self.norm_values = norm_values
        self.x_dim = 3

        
    def forward(self, z_data):

        molecule, protein_pocket = z_data

        # compute noised sample
        z_t_mol, z_t_pro, epsilon_mol, epsilon_pro, t = self.noise_process(z_data)

        # use neural network to predict noise
        epsilon_hat_mol, epsilon_hat_pro = self.neural_net(z_t_mol, z_t_pro, t, molecule['idx'], protein_pocket['idx'])

        # compute denoised sample
        # z_data_hat = (1 / alpha_t) * z_t - (sigma_t / alpha_t) * epsilon_hat

        ## Loss computation part (1) t != 0

        # compute the sum squared error loss per graph
        # TODO: set error_pro to 0 as pocket is fixed
        error_mol = scatter_add(torch.sum((epsilon_mol - epsilon_hat_mol)**2, dim=-1), molecule['idx'], dim=0)
        error_pro = scatter_add(torch.sum((epsilon_pro - epsilon_hat_pro)**2, dim=-1)**2, protein_pocket['idx'], dim=0)

        # additional evaluation (VLB) variables
        # TODO: add SNR weight computation
        SNR_weight = - 1
        # TODO: add delta_log_px (normalisation) and log_pN (t = 0)
        delta_log_px = 0
        log_pN = 0
        # TODO: add KL_prior loss (neglebile)
        kl_prior = 0
        # TODO: add neg_log_const
        neg_log_const = 0

        # TODO optional: can add auxiliary loss / lennard-jones potential

        # TODO: check all the gamma_t, alpha_t, sigma_t buissness again

        ## Loss computation part (2) t = 0

        # t = 0 and t != 0 masks for seperate computation of log p(x | z0)
        t_0_mask = (t == 0).float().squeeze()
        t_not_0_mask = 1 - t_0_mask

        if self.training:

            loss_x_mol_t0, loss_x_protein_t0, loss_h_t0 = self.loss_t0(
                molecule, z_t_mol, epsilon_mol, epsilon_hat_mol,
                protein_pocket, z_t_pro, epsilon_pro, epsilon_hat_pro, t
            )

            # seperate loss computation for t = 0 and t != 0
            loss_x_mol_t0 = - loss_x_mol_t0 * t_0_mask
            loss_x_protein_t0 = - loss_x_protein_t0 * t_0_mask
            loss_h_t0 = - loss_h_t0 * t_0_mask
            error_mol = error_mol * t_not_0_mask
            error_pro = error_pro * t_not_0_mask

        else:

            # For evaluation we want to compute t = 0 losses for all z_data samples that we have

            # compute noised sample for t = 0
            z_0_mol, z_0_pro, epsilon_0_mol, epsilon_0_pro, t_0 = self.noise_process(z_data, t_is_0 = True)

            # use neural network to predict noise for t = 0
            epsilon_hat_0_mol, epsilon_hat_0_pro = self.neural_net(z_0_mol, z_0_pro, t_0, molecule['idx'], protein_pocket['idx'])

            loss_x_mol_t0, loss_x_protein_t0, loss_h_t0 = self.loss_t0(
                molecule, z_0_mol, epsilon_0_mol, epsilon_hat_0_mol,
                protein_pocket, z_0_pro, epsilon_0_pro, epsilon_hat_0_pro, t_0
            )

            loss_x_mol_t0 = - loss_x_mol_t0
            loss_x_protein_t0 = - loss_x_protein_t0
            loss_h_t0 = - loss_h_t0

        ## Loss computation part (3) Normalisation
            
        # Loss terms: error_mol, error_pro, loss_x_mol_t0, loss_x_protein_t0, loss_h_t0
            
        # protein_pocket_fixed
        error_pro = 0
        loss_x_protein_t0 = 0
            
        if self.training:

            # Normalize loss_t by graph size
            error_mol = error_mol / ((molecule['x'].size(0) + self.num_atoms) * molecule['size'])
            error_pro = error_pro / ((protein_pocket['x'].size(0) + self.num_residues * protein_pocket['size']))
            loss_t = 0.5 * (error_mol + error_pro)

            # Normalize loss_0 by graph size
            loss_x_mol_t0 = loss_x_mol_t0 / molecule['x'].size(0) * molecule['size']
            loss_x_protein_t0 = loss_x_protein_t0 / protein_pocket['x'].size(0) * protein_pocket['size']
            loss_0 = loss_x_mol_t0 + loss_x_protein_t0 + loss_h_t0

            loss = loss_t + loss_0 + kl_prior

        else:

            # For evaluation we don't normalize ??? and compte vlb instead of l2 (currently vlb = l2)

            loss_t = - self.T * 0.5 * SNR_weight * (error_mol + error_pro)
            loss_0 = loss_x_mol_t0 + loss_x_protein_t0 + loss_h_t0
            loss_0 = loss_0 + neg_log_const

            # Two added loss terms for vlb
            loss = loss_t + loss_0 + kl_prior - delta_log_px - log_pN

        return loss.mean(0)

    def noise_process(self, z_data, t_is_0 = False):

        """
        Creates noised samples from data_samples following a predefined noise schedule
        """

        molecule, protein_pocket = z_data
        batch_size = molecule['size'].size(0)
        device = molecule['x'].device

        # normalisation with norm_values (dataset dependend) -> changes likelyhood (adjust vlb)!
        molecule['x'] = molecule['x'] / self.norm_values[0]
        molecule['h'] = molecule['h'] / self.norm_values[1]
        protein_pocket['x'] = protein_pocket['x'] / self.norm_values[0]
        protein_pocket['h'] = protein_pocket['h'] / self.norm_values[1]

        # sample t ~ U(0,...,T) for each graph individually
        t_low = 0 if self.train else 1
        t = torch.randint(t_low, self.T + 1, size=(batch_size, 1), device=device)
        # normalize t
        t = t / self.T

        # option for computing t = 0 representations
        t = torch.zeros((batch_size, 1), device=device) if t_is_0 else t
            
        # noise schedule
        alpha_t = 1 - (t / self.T)**2
        sigma_t = torch.sqrt(1 - alpha_t**2)


        # prepare joint point cloud
        xh_mol = torch.cat((molecule['x'], molecule['h']), dim=1)
        xh_pro = torch.cat((protein_pocket['x'], protein_pocket['h']), dim=1)
        idx_joint = torch.cat((molecule['idx'], protein_pocket['idx']))

        # center the input nodes
        xh_mol = scatter_mean(xh_mol, molecule['idx'], dim=0)[molecule['idx']]
        xh_pro = scatter_mean(xh_pro, protein_pocket['idx'], dim=0)[protein_pocket['idx']]

        # compute noised sample z_t
        # for x cord. we mean center the normal noise for each graph
        x_noise = torch.randn(size=(len(xh_mol) + len(xh_pro), self.x_dim), device=device)
        eps_x = x_noise - scatter_mean(x_noise, idx_joint, dim=0)[idx_joint]
        # for h we need standard normal noise
        # TODO: add option to turn this off for peptides
        eps_h_mol = torch.randn(size=(len(xh_mol), self.num_atoms), device=device)
        eps_h_pro = torch.randn(size=(len(xh_pro), self.num_residues), device=device)
        epsilon_mol = torch.cat((eps_x[:len(xh_mol)], eps_h_mol), dim=1)
        epsilon_pro = torch.cat((eps_x[len(xh_mol):], eps_h_pro), dim=1)

        # compute noised representations
        # alpha_t: [16,1] -> [300,13] or [333,23] by indexing and broadcasting
        z_t_mol = alpha_t[molecule['idx']] * xh_mol - sigma_t[molecule['idx']] * epsilon_mol
        z_t_pro = alpha_t[protein_pocket['idx']] * xh_pro - sigma_t[protein_pocket['idx']] * epsilon_pro

        return z_t_mol, z_t_pro, epsilon_mol, epsilon_pro, t
    
    def loss_t0(
            self, molecule, z_t_mol, epsilon_mol, epsilon_hat_mol,
            protein_pocket, z_t_pro, epsilon_pro, epsilon_hat_pro, 
            t, epsilon=1e-10
    ):
        """
        This function calculate log(p(xh|z_0))
        Replicated from [Schneuing et al. 2023]
        """

        device = molecule['x'].device

        ## Normal computation of position error (so t = 0 special case only important for h?)

        epsilon_mol_x = epsilon_mol[:,:self.x_dim]
        epsilon_hat_mol_x = epsilon_hat_mol[:,:self.x_dim]
        loss_x_mol_t0 = - 0.5 * scatter_add(torch.sum((epsilon_mol_x - epsilon_hat_mol_x)**2, dim=-1), molecule['idx'], dim=0)

        # TODO: if protein pocket not fixed add loss_x_protein_t0 computation
        loss_x_protein_t0 = 0

        ## Computation for changed features (so t = 0 special case only important for h?)

        # TODO: need to be careful that the shape is correct
        # TODO: need to check my custom sigma/alpha vs DiffDocks gamma
        alpha_0 = 1 - (t / self.T)**2
        sigma_0 = torch.sqrt(1 - alpha_0**2)
        sigma_0_unnormalized = sigma_0 * self.norm_values[1]
        # unnormalize not necessary for molecule['h'] because molecule was only locally normalized (can change that if necessary later)
        mol_h_hat = z_t_mol[:, self.x_dim:] * self.norm_values[1]
        mol_h_hat_centered = mol_h_hat - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        # N(mean=z_h_cat, stdev=sigma_0_cat)
        # 0.5 * (1. + torch.erf(x / math.sqrt(2)))
        log_probabilities_mol_unnormalized = torch.log(
            0.5 * (1. + torch.erf((mol_h_hat_centered + 0.5) / sigma_0_unnormalized[molecule['idx']]) / torch.sqrt(torch.Tensor([2]), device=device)) \
            - 0.5 * (1. + torch.erf((mol_h_hat_centered - 0.5) / sigma_0_unnormalized[molecule['idx']]) / torch.sqrt(torch.Tensor([2]), device=device)) \
            + epsilon
        )

        # Normalize the distribution over the categories.
        log_Z = torch.logsumexp(log_probabilities_mol_unnormalized, dim=1,
                                keepdim=True)
        
        log_probabilities_mol = log_probabilities_mol_unnormalized - log_Z

        loss_h_t0 = scatter_add(torch.sum(log_probabilities_mol * molecule['h'], dim=-1), molecule['idx'], dim=0)
        
        return loss_x_mol_t0, loss_x_protein_t0, loss_h_t0