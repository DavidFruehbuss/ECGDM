import torch
import numpy as np
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean

from Models.noise_schedule import Noise_Schedule
from Data.Peptide_data.utils import create_new_pdb

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
        protein_pocket_fixed: bool,
        features_fixed: bool,
        position_encoding: bool,
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
        self.protein_pocket_fixed = protein_pocket_fixed
        self.features_fixed = features_fixed
        self.position_encoding = position_encoding

        # dataset info
        self.num_atoms = num_atoms
        self.num_residues = num_residues
        self.norm_values = norm_values
        self.x_dim = 3

        # Noise Schedule
        self.noise_schedule = Noise_Schedule(self.T)

        self.com_old = False
        # This might be wrongly done, but is by far the simplest way to try this
        self.noise_scaling = 1
        
    def forward(self, z_data):

        molecule, protein_pocket = z_data
        if self.position_encoding:
            molecule_pos = molecule['pos_in_seq']
        else:
            molecule_pos = None

        # print(f'Molecule {molecule}')
        # print(f'protein_pocket {protein_pocket}')

        # computing the target
        mol_target = molecule['x'].detach().clone()
        # move to COM-0
        mol_target[:,:self.x_dim] = mol_target[:,:self.x_dim] - scatter_mean(mol_target[:,:self.x_dim], molecule['idx'], dim=0)[molecule['idx']]

        # print(f'mol_target {mol_target}')

        # compute noised sample
        z_t_mol, z_t_pro, epsilon_mol, epsilon_pro, t = self.noise_process(z_data)

        # print(f'z_t_mol {z_t_mol}')
        # print(f'z_t_pro {z_t_pro}')
        # print(f'epsilon_mol {epsilon_mol}')
        # print(f'epsilon_pro {epsilon_pro}')
        # print(f't {t}')

        # use neural network to predict noise
        epsilon_hat_mol, epsilon_hat_pro = self.neural_net(z_t_mol, z_t_pro, t, molecule['idx'], protein_pocket['idx'], molecule_pos)

        # print(f'epsilon_hat_mol {epsilon_hat_mol}')
        # print(f'epsilon_hat_pro {epsilon_hat_pro}')

        # print(f'epsilon_hat_mol.shape {epsilon_hat_mol.shape}')
        # print(f'epsilon_hat_pro.shape {epsilon_hat_pro.shape}')

        # compute alpha, sigma
        alpha_t = self.noise_schedule(t, 'alpha')
        sigma_t = self.noise_schedule(t, 'sigma')

        # print(f'alpha_t {alpha_t}')
        # print(f'sigma_t {sigma_t}')

        # compute denoised sample
        # original equation
        z_data_hat = (1 / alpha_t)[molecule['idx']] * z_t_mol - (sigma_t / alpha_t)[molecule['idx']] * epsilon_hat_mol

        # print(f'z_data_hat {z_data_hat}')

        # above is revers of this equation
        # z_t_mol = alpha_t[molecule['idx']] * xh_mol + sigma_t[molecule['idx']] * epsilon_mol

        # Log sampling progress to wandb
        error_mol = scatter_add(torch.sum((mol_target - z_data_hat[:,:3])**2, dim=-1), molecule['idx'], dim=0)
        rmse = torch.sqrt(error_mol / (3 * molecule['size']))


        if self.training:

            loss, info = self.train_loss(molecule, z_t_mol, epsilon_mol, 
                                         epsilon_hat_mol,protein_pocket, 
                                         z_t_pro, epsilon_pro, epsilon_hat_pro, t)

        else: 

            loss, info = self.validation_loss(z_data, molecule, z_t_mol, epsilon_mol, 
                                         epsilon_hat_mol,protein_pocket, 
                                         z_t_pro, epsilon_pro, epsilon_hat_pro, t)
            
        info['rmse'] = rmse.mean(0)

        return loss.mean(0), info

    def noise_process(self, z_data, t_is_0 = False):

        """
        Creates noised samples from data_samples following a predefined noise schedule
        - Entire operation takes place with center of gravity = 0
        """

        molecule, protein_pocket = z_data
        batch_size = molecule['size'].size(0)
        device = molecule['x'].device

        # normalisation with norm_values (dataset dependend) -> changes likelyhood (adjusted for in vlb)!
        molecule['x'] = molecule['x'] / self.norm_values[0]
        molecule['h'] = molecule['h'] / self.norm_values[1]
        protein_pocket['x'] = protein_pocket['x'] / self.norm_values[0]
        protein_pocket['h'] = protein_pocket['h'] / self.norm_values[1]

        # print(f'molecule_xh {molecule}')
        # print(f'protein_pocket_xh {protein_pocket}')

        # sample t ~ U(0,...,T) for each graph individually
        t_low = 0 if self.train else 1

        # max_T = int(self.T * 0.890) # modifier to only train up to a certain max_noise value
        max_T = self.T
        t = torch.randint(t_low, max_T + 1, size=(batch_size, 1), device=device)

        ## TODO: Fix t for comparssion
        t[:,:] = 100

        # high_noise_training_schedule
        self.high_noise_training_schedule = False
        if self.high_noise_training_schedule:
            split_point = int(0.95 * self.T)
            t = torch.empty((batch_size, 1), device=device)
            for i in range(batch_size):
                rnd = torch.rand(1, device=device)
                if rnd > 0.5:
                    # Sample from the lower segment (t_low to 80% of T)
                    t[i] = torch.randint(t_low, split_point + 1, (1,), device=device)
                else:
                    # Sample from the upper segment (80% of T to T)
                    t[i] = torch.randint(split_point + 1, self.T + 1, (1,), device=device)


        # normalize t
        t = t / self.T

        # option for computing t = 0 representations
        t = torch.zeros((batch_size, 1), device=device) if t_is_0 else t
            
        # noise schedule
        alpha_t = self.noise_schedule(t, 'alpha')
        sigma_t = self.noise_schedule(t, 'sigma')

        # prepare joint point cloud
        xh_mol = torch.cat((molecule['x'], molecule['h']), dim=1)
        xh_pro = torch.cat((protein_pocket['x'], protein_pocket['h']), dim=1)

        if self.protein_pocket_fixed:

            if self.com_old:
                # old centering approach
                xh_mol[:,:self.x_dim] = xh_mol[:,:self.x_dim] - scatter_mean(xh_mol[:,:self.x_dim], molecule['idx'], dim=0)[molecule['idx']]
                xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - scatter_mean(xh_pro[:,:self.x_dim], protein_pocket['idx'], dim=0)[protein_pocket['idx']]
            else:
                # data is translated to 0, COM noise added and again translated to 0
                mean = scatter_mean(xh_mol[:,:self.x_dim], molecule['idx'], dim=0)
                xh_mol[:,:self.x_dim] = xh_mol[:,:self.x_dim] - mean[molecule['idx']]
                xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - mean[protein_pocket['idx']]

                # print(f'xh_mol after mean {xh_mol}')
                # print(f'xh_pro after mean {xh_pro}')

            # compute noised sample z_t
            # for x cord. we mean center the normal noise for each graph
            # modify to only diffuse position of the molecule
            eps_x_mol = torch.randn(size=(len(xh_mol), self.x_dim), device=device) * self.noise_scaling
            eps_x_pro = torch.zeros(size=(len(xh_pro), self.x_dim), device=device)

            if self.com_old:
                # old centering approach
                eps_x_mol = eps_x_mol - scatter_mean(eps_x_mol, molecule['idx'], dim=0)[molecule['idx']]
                eps_x_pro = torch.zeros(size=(len(xh_pro), self.x_dim), device=device)

            if self.features_fixed:

                eps_h_mol = torch.zeros(size=(len(xh_mol), self.num_atoms), device=device)
                eps_h_pro = torch.zeros(size=(len(xh_pro), self.num_residues), device=device)

            else:
                # for h we need standard normal noise
                eps_h_mol = torch.randn(size=(len(xh_mol), self.num_atoms), device=device)
                eps_h_pro = torch.randn(size=(len(xh_pro), self.num_residues), device=device)

            epsilon_mol = torch.cat((eps_x_mol, eps_h_mol), dim=1)
            epsilon_pro = torch.cat((eps_x_pro, eps_h_pro), dim=1)

            # compute noised representations
            # alpha_t: [16,1] -> [300,13] or [333,23] by indexing and broadcasting
            # TODO: alpha value confuses me, doesn't this change the size of the complex (unstable variance)
            # TODO:1 - instead of + ????
            z_t_mol = alpha_t[molecule['idx']] * xh_mol + sigma_t[molecule['idx']] * epsilon_mol
            # TODO: massive change [alternative is to update pocket in sampling as well]
            z_t_pro = xh_pro
            # z_t_pro = alpha_t[protein_pocket['idx']] * xh_pro + sigma_t[protein_pocket['idx']] * epsilon_pro

            if self.com_old:
                dumy_variable = 0
            else:
                # data is translated to 0, COM noise added and again translated to 0 (turn off for old centering approach)
                mean = scatter_mean(z_t_mol[:,:self.x_dim], molecule['idx'], dim=0)
                z_t_mol[:,:self.x_dim] = z_t_mol[:,:self.x_dim] - mean[molecule['idx']]
                z_t_pro[:,:self.x_dim] = z_t_pro[:,:self.x_dim] - mean[protein_pocket['idx']]

        else:
            # COM modification if protein_pocket not fixed
            # noise sampling is different
            # data_position stays and noise is translated
            raise NotImplementedError

        return z_t_mol, z_t_pro, epsilon_mol, epsilon_pro, t
    
    def train_loss(
            self, molecule, z_t_mol, epsilon_mol, epsilon_hat_mol,
            protein_pocket, z_t_pro, epsilon_pro, epsilon_hat_pro, 
            t,
    ):
        # compute the sum squared error loss per graph # TODO: modified to not take the h_dims
        error_mol = scatter_add(torch.sum((epsilon_mol[:,:3] - epsilon_hat_mol[:,:3])**2, dim=-1), molecule['idx'], dim=0)

        if self.protein_pocket_fixed:
            error_pro = torch.zeros(protein_pocket['size'].size(0), device=molecule['x'].device)
        else:
            error_pro = scatter_add(torch.sum((epsilon_pro - epsilon_hat_pro)**2, dim=-1), protein_pocket['idx'], dim=0)

        # TODO: add KL_prior loss (neglebile) <------ might be not neglebile after all
        kl_prior = self.kl_prior(molecule)

        # Add a SNR modulation term to upweight highly noised samples
        SNR_t = (1 / self.SNR_t(t).squeeze(1))

        # t = 0 and t != 0 masks for seperate computation of log p(x | z0)
        t_0_mask = (t == 0).float().squeeze()
        t_not_0_mask = 1 - t_0_mask

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

        # Normalize loss_t by graph size
        error_mol = error_mol / ((self.x_dim) * molecule['size'])
        error_pro = error_pro / ((self.x_dim + self.num_residues * protein_pocket['size']))
        loss_t = 0.5 * (error_mol + error_pro) # * SNR_t

        # Normalize loss_0 by graph size
        loss_x_mol_t0 = loss_x_mol_t0 / (self.x_dim * molecule['size'])
        loss_x_protein_t0 = loss_x_protein_t0 / (self.x_dim * protein_pocket['size'])
        loss_0 = loss_x_mol_t0 + loss_x_protein_t0 + loss_h_t0

        loss = loss_t + loss_0 + kl_prior

        info = {
            'loss_t': loss_t.mean(0),
            'loss_0': loss_0.mean(0),
            'error_mol': error_mol.mean(0),
            'error_pro': error_pro.mean(0),
            'loss_x_mol_t0': loss_x_mol_t0.mean(0),
            'loss_x_protein_t0': loss_x_protein_t0.mean(0),
            'loss_h_t0': loss_h_t0.mean(0),
            'kl_prior': kl_prior.mean(0),
        }

        return loss, info
    
    def validation_loss(
            self, z_data, molecule, z_t_mol, epsilon_mol, epsilon_hat_mol,
            protein_pocket, z_t_pro, epsilon_pro, epsilon_hat_pro, 
            t,
    ):
        ### Additional evaluation (VLB) variables
        if self.position_encoding:
            molecule_pos = molecule['pos_in_seq']
        else:
            molecule_pos = None

        # compute the sum squared error loss per graph
        error_mol = scatter_add(torch.sum((epsilon_mol[:,:3] - epsilon_hat_mol[:,:3])**2, dim=-1), molecule['idx'], dim=0)

        if self.protein_pocket_fixed:
            error_pro = torch.zeros(protein_pocket['size'].size(0), device=molecule['x'].device)
        else:
            error_pro = scatter_add(torch.sum((epsilon_pro - epsilon_hat_pro)**2, dim=-1), protein_pocket['idx'], dim=0)

        # TODO: add KL_prior loss (neglebile)
        kl_prior = self.kl_prior(molecule)

        # if pocket not fixed then molecule['size'] + protein_pocket['size']
        neg_log_const = self.neg_log_const(molecule['size'], molecule['size'].size(0), device=molecule['x'].device)
        delta_log_px = self.delta_log_px(molecule['size'])
        # SNR is computed between timestep s and t (with s = t-1)
        SNR_weight = (1 - self.SNR_s_t(t).squeeze(1))

        # TODO: add log_pN computation using the dataset histogram
        log_pN = self.log_pN(molecule['size'], protein_pocket['size'])

        # TODO optional: can add auxiliary loss / lennard-jones potential

        ## For evaluation we want to compute t = 0 losses for all z_data samples that we have

        # compute noised sample for t = 0
        z_0_mol, z_0_pro, epsilon_0_mol, epsilon_0_pro, t_0 = self.noise_process(z_data, t_is_0 = True)

        # use neural network to predict noise for t = 0
        epsilon_hat_0_mol, epsilon_hat_0_pro = self.neural_net(z_0_mol, z_0_pro, t_0, molecule['idx'], protein_pocket['idx'], molecule_pos)

        loss_x_mol_t0, loss_x_protein_t0, loss_h_t0 = self.loss_t0(
            molecule, z_0_mol, epsilon_0_mol, epsilon_hat_0_mol,
            protein_pocket, z_0_pro, epsilon_0_pro, epsilon_hat_0_pro, t_0
        )

        loss_x_mol_t0 = - loss_x_mol_t0
        loss_x_protein_t0 = - loss_x_protein_t0
        loss_h_t0 = - loss_h_t0

        ## For evaluation we don't normalize ??? and compte vlb instead of l2 (currently vlb = l2)

        loss_t = - self.T * 0.5 * SNR_weight * (error_mol + error_pro)
        loss_0 = loss_x_mol_t0 + loss_x_protein_t0 + loss_h_t0
        loss_0 = loss_0 + neg_log_const

        # Two added loss terms for vlb
        loss = loss_t + loss_0 + kl_prior - delta_log_px - log_pN

        info = {
            'loss_t': loss_t.mean(0),
            'loss_0': loss_0.mean(0),
            'error_mol': error_mol.mean(0),
            'error_pro': error_pro.mean(0),
            'loss_x_mol_t0': loss_x_mol_t0.mean(0),
            'loss_x_protein_t0': loss_x_protein_t0.mean(0),
            'loss_h_t0': loss_h_t0.mean(0),
            'kl_prior': kl_prior.mean(0),
            'neg_log_const': neg_log_const.mean(0),
            'delta_log_px': delta_log_px.mean(0),
            'log_pN': log_pN,
            'SNR_weight': SNR_weight.mean(0)
        }

        return loss, info
        
    
    def loss_t0(
            self, molecule, z_t_mol, epsilon_mol, epsilon_hat_mol,
            protein_pocket, z_t_pro, epsilon_pro, epsilon_hat_pro, 
            t, epsilon=1e-10
    ):
        """
        This function calculate log(p(xh|z_0))
        Replicated from [Schneuing et al. 2023]
        """

        ## Normal computation of position error (so t = 0 special case only important for h?)

        epsilon_mol_x = epsilon_mol[:,:self.x_dim]
        epsilon_hat_mol_x = epsilon_hat_mol[:,:self.x_dim]
        loss_x_mol_t0 = - 0.5 * scatter_add(torch.sum((epsilon_mol_x - epsilon_hat_mol_x)**2, dim=-1), molecule['idx'], dim=0)

        if self.protein_pocket_fixed:
            loss_x_protein_t0 = torch.zeros(protein_pocket['size'].size(0), device=molecule['x'].device)
        else:
            raise NotImplementedError

        if self.features_fixed:

            loss_h_t0 = torch.zeros(molecule['size'].size(0), device=molecule['x'].device)

        else:

            ## Computation for changed features (so t = 0 special case only important for h?)

            # TODO: value here is about 100 times to big
            sigma_0 = self.noise_schedule(t, 'sigma')
            sigma_0_unnormalized = sigma_0 * self.norm_values[1]
            # unnormalize not necessary for molecule['h'] because molecule was only locally normalized (can change that if necessary later)
            mol_h_hat = z_t_mol[:, self.x_dim:] * self.norm_values[1]
            mol_h_hat_centered = mol_h_hat - 1

            # Compute integrals from 0.5 to 1.5 of the normal distribution
            # N(mean=z_h_cat, stdev=sigma_0_cat)
            # 0.5 * (1. + torch.erf(x / math.sqrt(2)))
            log_probabilities_mol_unnormalized = torch.log(
                0.5 * (1. + torch.erf((mol_h_hat_centered + 0.5) / sigma_0_unnormalized[molecule['idx']]) / math.sqrt(2)) \
                - 0.5 * (1. + torch.erf((mol_h_hat_centered - 0.5) / sigma_0_unnormalized[molecule['idx']]) / math.sqrt(2)) \
                + epsilon
            )

            # Normalize the distribution over the categories.
            log_Z = torch.logsumexp(log_probabilities_mol_unnormalized, dim=1,
                                    keepdim=True)
            
            log_probabilities_mol = log_probabilities_mol_unnormalized - log_Z

            loss_h_t0 = scatter_add(torch.sum(log_probabilities_mol * molecule['h'], dim=-1), molecule['idx'], dim=0)
        
        return loss_x_mol_t0, loss_x_protein_t0, loss_h_t0
    
    def kl_prior(self, molecule):

        device=molecule['x'].device

        molecule['x'] = molecule['x'] - scatter_mean(molecule['x'], molecule['idx'], dim=0)[molecule['idx']]

        T_normalized = torch.ones((len(molecule['size']), 1), device=device)
        alpha_T = self.noise_schedule(T_normalized, 'alpha')
        sigma_T = self.noise_schedule(T_normalized, 'sigma')
        sigma_T_value = sigma_T[0,0].item()

        mu_x_mol = molecule['x'] * alpha_T[molecule['idx']] # [:,3]
        mu_h_mol = molecule['h'] * alpha_T[molecule['idx']] # [:,20]

        sigma_T_x = torch.full(alpha_T.shape, fill_value=sigma_T_value, device=device).squeeze() # [64,1]
        sigma_T_h = torch.full(alpha_T.shape, fill_value=sigma_T_value, device=device).squeeze() # [64,1]

        # KL computation h
        kl_h = 0
        # zeros = torch.zeros_like(mu_h_mol)
        # ones = torch.ones_like(sigma_T_h)
        # mu_norm2 = scatter_add(torch.sum((mu_h_mol - zeros) ** 2, dim=1), molecule['idx'], dim=0)
        # kl_h = torch.log(ones / sigma_T_h) + 0.5 * (sigma_T_h**2 + mu_norm2) / (ones**2) - 0.5

        # KL computation x
        zeros = torch.zeros_like(mu_x_mol)
        ones = torch.ones_like(sigma_T_x) * self.noise_scaling
        mu_norm2 = scatter_add(torch.sum((mu_x_mol - zeros) ** 2, dim=-1), molecule['idx'], dim=0)
        d = (molecule['size'] - 1) * self.x_dim
        kl_x = d * torch.log(ones / sigma_T_x) + 0.5 * (d * sigma_T_x**2 + mu_norm2) / (ones**2) - 0.5 * d

        kl_loss = kl_x + kl_h

        return kl_loss
    
    def delta_log_px(self, num_nodes):

        delta_log_px = - (num_nodes - 1) * self.x_dim * np.log(self.norm_values[0])

        return delta_log_px
    
    def log_pN(self, molecule_N, protein_pocket_N):

        # add log_pN computation using the dataset histogram
        log_pN = 0

        return log_pN
    
    def neg_log_const(self, num_nodes, batch_size, device):

        t0 = torch.zeros((batch_size, 1), device=device)
        log_sigma_0 = torch.log(self.noise_schedule(t0, 'sigma')).view(batch_size)

        neg_log_const = - ((num_nodes - 1) * self.x_dim) * (- log_sigma_0 - 0.5 * np.log(2 * np.pi))

        return neg_log_const
    
    def SNR_s_t(self, t):

        '''
        computes the SNR between t and the previous timestep t-1
        (why not t and 0?)
        '''

        s = torch.round(t * self.T).long() - 1
        s = s / self.T

        alpha2_t = self.noise_schedule(t, 'alpha')**2
        alpha2_s = self.noise_schedule(s, 'alpha')**2
        sigma2_t = self.noise_schedule(t, 'sigma')**2
        sigma2_s = self.noise_schedule(s, 'sigma')**2

        SNR_s_t = (alpha2_s / alpha2_t) / (sigma2_s / sigma2_t)

        return SNR_s_t
    
    def SNR_t(self, t):

        '''
        computes the SNR between t and 0
        '''

        alpha2_t = self.noise_schedule(t, 'alpha')**2
        sigma2_t = self.noise_schedule(t, 'sigma')**2

        SNR_t = alpha2_t / sigma2_t

        return SNR_t
    
    @torch.no_grad()
    def sample_structure(
            self,
            num_samples,
            molecule,
            protein_pocket,
            wandb,
            run_id,
        ):
        '''
        This function takes a molecule and a protein and return the most likely joint structure.
        - only structure sampling !
        Instead of giving a batch of molecule-protein pairs, we give a batch of identical replicates
        of the same pair, with the batch_size being the number of samples we want to generate.
        '''

        if self.protein_pocket_fixed == False:
            # for this the sampling would change due to COM trick
            raise NotImplementedError

        # replicate (molecule + protein_pocket) to have a batch of num_samples many replicates
        # do this step with Dataset function in lightning_modules
        device = molecule['x'].device
        if self.position_encoding:
            molecule_pos = molecule['pos_in_seq']
        else:
            molecule_pos = None
        num_samples = len(molecule['size'])

        # Record protein_pocket center of mass before
        protein_pocket_com_before = scatter_mean(protein_pocket['x'], protein_pocket['idx'], dim=0)

        # Presteps

        # If we want a new peptide we would have to add random sampling of the start peptide here^

        # define the target at the pocket position
        mol_target_p = molecule['x'] - scatter_mean(molecule['x'], molecule['idx'], dim=0)[molecule['idx']]
        mol_target_p = mol_target_p + protein_pocket_com_before[molecule['idx']]

        # define the target at 0-COM
        mol_target_0 = molecule['x'] - scatter_mean(molecule['x'], molecule['idx'], dim=0)[molecule['idx']]

        # Normalisation
        # molecule['x'] = molecule['x'] / self.norm_values[0]
        molecule['h'] = molecule['h'] / self.norm_values[1]
        protein_pocket['x'] = protein_pocket['x'] / self.norm_values[0]
        protein_pocket['h'] = protein_pocket['h'] / self.norm_values[1]

        # start with random peptide position (target hidden)
        # mean=COM, sigma=1, and sample epsioln
        rand_eps_x = torch.randn((len(molecule['x']), self.x_dim), device=device) * self.noise_scaling

        rand_eps_x = torch.tensor([[ 0.2907,  0.2309,  1.2472],
                                    [ 0.4251,  0.1150, -1.0757],
                                    [-0.4606,  0.9788, -0.1525],
                                    [-0.2532, -0.6566,  0.1485],
                                    [ 0.2235,  0.0785,  0.6607],
                                    [-0.2066, -0.2605,  0.2740],
                                    [-0.7848, -1.1916, -0.6044],
                                    [ 1.4707,  0.6776, -1.0164],
                                    [-0.7048,  0.0280,  0.5186]], device='cuda:0')

        molecule_x = protein_pocket_com_before[molecule['idx']] + rand_eps_x

        # TODO: here features get changed! (for peptides need to turn this off somehow)
        if self.features_fixed:
            molecule_h = molecule['h'].clone().detach()
        else:
            # run moad sampling with only structure diffusion
            molecule_h = molecule['h'].clone().detach()
            # molecule_h = torch.zeros((protein_pocket['h'].size), device=device)

        # combine position and features
        xh_mol = torch.cat((molecule_x, molecule_h), dim=1)
        xh_pro = torch.cat((protein_pocket['x'], protein_pocket['h']), dim=1)

        error_mol = scatter_add(torch.sum((mol_target_p - xh_mol[:,:3])**2, dim=-1), molecule['idx'], dim=0)
        rmse = torch.sqrt(error_mol / (3 * molecule['size']))
        print(f'The starting RSME of random noise is {rmse.mean(0)}')

        if self.com_old:
            # old centering approach
            xh_mol[:,:self.x_dim] = xh_mol[:,:self.x_dim] - scatter_mean(xh_mol[:,:self.x_dim], molecule['idx'], dim=0)[molecule['idx']]
            xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - scatter_mean(xh_pro[:,:self.x_dim], protein_pocket['idx'], dim=0)[protein_pocket['idx']]
        else:
            # data is translated to 0, COM noise added and again translated to 0
            mean = scatter_mean(xh_mol[:,:self.x_dim], molecule['idx'], dim=0)
            xh_mol[:,:self.x_dim] = xh_mol[:,:self.x_dim] - mean[molecule['idx']]
            xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - mean[protein_pocket['idx']]

        error_mol = scatter_add(torch.sum((mol_target_0 - xh_mol[:,:3])**2, dim=-1), molecule['idx'], dim=0)
        rmse = torch.sqrt(error_mol / (3 * molecule['size']))
        print(f'The starting RSME of random noise at COM 0 is {rmse.mean(0)}')

        # # Sanaty check
        molecule_xx = molecule['x'].clone().detach()
        molecule_xx[:,:self.x_dim] = molecule_xx[:,:self.x_dim] - scatter_mean(molecule_xx[:,:self.x_dim], molecule['idx'], dim=0)[molecule['idx']]
        xh_t_mol = torch.cat((molecule_xx, molecule_h), dim=1)
        z_t_pro = xh_pro.clone().detach()

        error_mol = scatter_add(torch.sum((mol_target_0 - molecule_xx[:,:3])**2, dim=-1), molecule['idx'], dim=0)
        rmse = torch.sqrt(error_mol / (3 * molecule['size']))
        print(f'Sanity check 1 (self): {rmse.mean(0)}')

        # # epsilon for the sanity checks
        # eps_x_mol = torch.randn(size=(len(xh_mol), self.x_dim), device=device) * self.noise_scaling

        # if self.com_old:
        #     # old centering approach
        #     eps_x_mol = eps_x_mol - scatter_mean(eps_x_mol, molecule['idx'], dim=0)[molecule['idx']]

        # if self.features_fixed:
        #     eps_h_mol = torch.zeros(size=(len(xh_mol), self.num_atoms), device=device)

        # epsilon_mol = torch.cat((eps_x_mol, eps_h_mol), dim=1)

        if self.T == 1000:
            TS = [1, 50, 100, 200, 400, 500, 700, 800, 900, 950, 998]
        else:
            TS = [1, 2, 10, 25, 50, 100, 200, 300, 400, 450, 498]

        # for ts in TS:

        #     t_array = torch.randint(ts, ts + 1, size=(num_samples, 1), device=device)
        #     # normalize t
        #     t_array = t_array / self.T
        #     alpha_t = self.noise_schedule(t_array, 'alpha')
        #     sigma_t = self.noise_schedule(t_array, 'sigma')

        #     z_t_mol = alpha_t[molecule['idx']] * xh_t_mol + sigma_t[molecule['idx']] * epsilon_mol

        #     self.safe_pdbs(z_t_mol, molecule, run_id, time_step=f'N_{ts}')

        #     # use neural network to predict noise
        #     epsilon_hat_mol, epsilon_hat_pro = self.neural_net(z_t_mol, z_t_pro, t_array, molecule['idx'], protein_pocket['idx'], molecule_pos)

        #     # compute denoised sample
        #     # original equation
        #     z_data_hat = (1 / alpha_t)[molecule['idx']] * z_t_mol - (sigma_t / alpha_t)[molecule['idx']] * epsilon_hat_mol
        #     z_data_hat_2 = (1 / alpha_t)[molecule['idx']] * z_t_mol - (sigma_t / alpha_t)[molecule['idx']] * epsilon_mol

        #     # expanend_fraction = (sigma_t / alpha_t)[molecule['idx']]
        #     # print(f'sigma {sigma_t.shape}')
        #     # print(f'alpha {alpha_t.shape}')
        #     # print(f'sigma/alpha {(sigma_t / alpha_t).shape}')
        #     # print(f'alpha[mol_idx] {expanend_fraction.shape}')

        #     error_mol = scatter_add(torch.sum((molecule_xx - z_data_hat[:,:3])**2, dim=-1), molecule['idx'], dim=0)
        #     rmse = torch.sqrt(error_mol / (3 * molecule['size']))
        #     print(f'Sanity Check 2 (normal) {rmse.mean(0)}')

        #     error_mol = scatter_add(torch.sum((molecule_xx - z_data_hat_2[:,:3])**2, dim=-1), molecule['idx'], dim=0)
        #     rmse = torch.sqrt(error_mol / (3 * molecule['size']))
        #     print(f'Sanity Check 3 (true noise) {rmse.mean(0)}')

        #     self.safe_pdbs(z_data_hat, molecule, run_id, time_step=f'DN{ts}')

        # # Visualize true peptide pmhc
        # self.safe_pdbs(z_data_hat_2, molecule, run_id, time_step=f'T{ts}')

        max_T = self.T

        ##################
        # Sanity Check with ts steps from ts/1000 noised sample
        # ts = 950
        # t_array = torch.randint(ts, ts + 1, size=(num_samples, 1), device=device)
        # # normalize t
        # t_array = t_array / self.T
        # alpha_t = self.noise_schedule(t_array, 'alpha')
        # sigma_t = self.noise_schedule(t_array, 'sigma')

        # z_t_mol = alpha_t[molecule['idx']] * xh_t_mol + sigma_t[molecule['idx']] * epsilon_mol

        # self.safe_pdbs(z_t_mol, molecule, run_id, time_step=f'SDNS_{ts}')

        # xh_mol = z_t_mol
        # max_T = ts
        ##################

        print(f'x_mol_before {xh_mol}')
        print(f'x_pro_before {xh_pro}')


        # Iterativly denoise stepwise for t = T,...,1
        for s in reversed(range(0,max_T)):

            if s < max_T-2: raise NameError

            if s % 100 == 0 or s > 990:
                self.safe_pdbs(xh_mol, molecule, run_id, time_step=s)

            # time arrays
            s_array = torch.full((num_samples, 1), fill_value=s, device=device)
            t_array = s_array + 1
            s_array_norm = s_array / self.T
            t_array_norm = t_array / self.T

            # compute alpha_s, alpha_t, sigma_s, sigma_t, alpha_t_given_s, sigma_t_given_s & sigma2_t_given_s
            # alpha_t_given_s = alpha_t / alpha_s, sigma_t_given_s = sqrt(1 - (alpha_t_given_s) ^2 )
            alpha_s = self.noise_schedule(s_array_norm, 'alpha')
            alpha_t = self.noise_schedule(t_array_norm, 'alpha')
            sigma_s = self.noise_schedule(s_array_norm, 'sigma')
            sigma_t = self.noise_schedule(t_array_norm, 'sigma')
            alpha_t_given_s = alpha_t / alpha_s
            sigma_t_given_s = torch.sqrt(1 - (alpha_t_given_s)**2 )
            sigma2_t_given_s = sigma_t_given_s**2

            # use neural network to predict noise
            epsilon_hat_mol, _ = self.neural_net(xh_mol, xh_pro, t_array_norm, molecule['idx'], protein_pocket['idx'], molecule_pos)

            # extra check
            # z_data_hat = (1 / alpha_t)[molecule['idx']] * xh_mol - (sigma_t / alpha_t)[molecule['idx']] * epsilon_hat_mol
            # z_data_hat[:,:self.x_dim] = z_data_hat[:,:self.x_dim] - scatter_mean(z_data_hat[:,:self.x_dim], molecule['idx'], dim=0)[molecule['idx']]
            # error_mol_check = scatter_add(torch.sqrt(torch.sum((mol_target_0 - z_data_hat[:,:3])**2, dim=-1)), molecule['idx'], dim=0)
            # rmse_check = error_mol_check / ((3 + self.num_atoms) * molecule['size'])
            # print(f'Direct denoising gives {rmse_check.mean(0)}')

            # compute p(z_s|z_t) using epsilon and alpha_s_given_t, sigma_s_given_t to predict mean and std of z_s
            mean_mol_s = xh_mol / alpha_t_given_s[molecule['idx']] - (sigma2_t_given_s / alpha_t_given_s / sigma_t)[molecule['idx']] * epsilon_hat_mol
            sigma_mol_s = sigma_t_given_s * sigma_s / sigma_t
            eps_mol_random = torch.randn(size=(len(xh_mol), self.x_dim + self.num_atoms), device=device) * self.noise_scaling # Here noise scaling might not be needed
            eps_mol_random = eps_mol_random - scatter_mean(eps_mol_random, molecule['idx'], dim=0)[molecule['idx']]
            # the line bellow is where we would add the gaudi guidance (-> compute backbone and statistical potentials)
            xh_mol = mean_mol_s + sigma_mol_s[molecule['idx']] * eps_mol_random
            xh_pro = xh_pro.detach().clone() # for safety (probally not necessary)

            self.sampling_without_noise = False
            if self.sampling_without_noise:
                xh_mol = mean_mol_s.clone().detach()

            if self.com_old:
                dumy_variable = 0
            else:
                # project both pocket and peptide to 0 COM again (only mol mean changes)
                mean = scatter_mean(xh_mol[:,:self.x_dim], molecule['idx'], dim=0)
                xh_mol[:,:self.x_dim] = xh_mol[:,:self.x_dim] - mean[molecule['idx']]
                xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - mean[protein_pocket['idx']]

            if self.com_old:
                # old centering approach
                xh_mol[:,:self.x_dim] = xh_mol[:,:self.x_dim] - scatter_mean(xh_mol[:,:self.x_dim], molecule['idx'], dim=0)[molecule['idx']]
                xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - scatter_mean(xh_pro[:,:self.x_dim], protein_pocket['idx'], dim=0)[protein_pocket['idx']]
            else:
                dumy_variable = 0

            # Log sampling progress
            error_mol = scatter_add(torch.sum((mol_target_0 - xh_mol[:,:3])**2, dim=-1), molecule['idx'], dim=0)
            rmse = torch.sqrt(error_mol / (3 * molecule['size']))
            print(rmse.mean(0))
            # wandb.log({'RMSE now': rmse.mean(0).item()})

        # sample final molecules with t = 0 (p(x|z_0)) [all the above steps but for t = 0]
        t_0_array_norm = torch.zeros((num_samples, 1), device=device)
        alpha_0 = self.noise_schedule(t_0_array_norm, 'alpha')
        sigma_0 = self.noise_schedule(t_0_array_norm, 'sigma')

        # use neural network to predict noise
        epsilon_hat_mol_0, _ = self.neural_net(xh_mol, xh_pro, t_0_array_norm, molecule['idx'], protein_pocket['idx'], molecule_pos)

        # compute p(x|z_0) using epsilon and alpha_0, sigma_0 to predict mean and std of x
        mean_mol_final = 1. / alpha_0[molecule['idx']] * (xh_mol - sigma_0[molecule['idx']] * epsilon_hat_mol_0)
        sigma_mol_final = sigma_0 / alpha_0 # not sure about this one
        eps_lig_random = torch.randn(size=(len(xh_mol), self.x_dim + self.num_atoms), device=device) * self.noise_scaling # Here noise scaling might not be needed
        xh_mol_final = mean_mol_final + sigma_mol_final[molecule['idx']] * eps_lig_random
        xh_pro_final = xh_pro.detach().clone() # for safety (probally not necessary)

        if self.com_old:
                dumy_variable = 0
        else:
            # project both pocket and peptide to 0 COM again (only mol mean changes)
            mean = scatter_mean(xh_mol_final[:,:self.x_dim], molecule['idx'], dim=0)
            xh_mol_final[:,:self.x_dim] = xh_mol_final[:,:self.x_dim] - mean[molecule['idx']]
            xh_pro_final[:,:self.x_dim] = xh_pro_final[:,:self.x_dim] - mean[protein_pocket['idx']]

        # Unnormalisation
        x_mol_final = xh_mol_final[:,:self.x_dim] * self.norm_values[0]
        h_mol_final = xh_mol_final[:,self.x_dim:] * self.norm_values[0]
        x_pro_final = xh_pro_final[:,:self.x_dim] * self.norm_values[0]
        h_pro_final = xh_pro_final[:,self.x_dim:] * self.norm_values[0]

        # Round h to one_hot encoding
        h_mol_final = F.one_hot(torch.argmax(h_mol_final, dim=1), self.num_atoms)

        # Recombine x and h
        xh_mol_final = torch.cat([x_mol_final, h_mol_final], dim=1)
        xh_pro_final = torch.cat([x_pro_final, h_pro_final], dim=1)

        # Correct for center of mass difference
        protein_pocket_com_after = scatter_mean(x_pro_final, protein_pocket['idx'], dim=0)

        # Testing if we only learn the form
        # Moving mol targets COM to 0
        mol_target = molecule['x']  - scatter_mean(molecule['x'], molecule['idx'], dim=0)[molecule['idx']]

        xh_mol_final[:,:self.x_dim] += (protein_pocket_com_before - protein_pocket_com_after)[molecule['idx']]
        xh_pro_final[:,:self.x_dim] += (protein_pocket_com_before - protein_pocket_com_after)[protein_pocket['idx']]

        # Moving mol targets COM to original COM
        mol_target += (protein_pocket_com_before - protein_pocket_com_after)[molecule['idx']]

        # Log sampling progress
        error_mol = scatter_add(torch.sum((mol_target - xh_mol_final[:,:3])**2, dim=-1), molecule['idx'], dim=0)
        rmse = torch.sqrt(error_mol / (3 * molecule['size']))
        print(f'Final RSME: {rmse.mean(0)}')
        # wandb.log({'RMSE now': rmse.mean(0).item()})

        sampled_structures = (xh_mol_final, xh_pro_final)

        self.safe_pdbs(xh_mol_final, molecule, run_id, time_step='F')
        

        return sampled_structures
    
    def safe_pdbs(self, pos, molecule, run_id, time_step):

        return

        if not self.features_fixed:
            # idicates ligand (not peptide) dataset
            return

        for i in range(len(molecule['size'])):
            # (1) extract the peptide position
            pos = pos[:,:3]
            peptide_pos = pos[molecule['idx'] == i]
            # (2) bring peptides into correct order
            peptide_idx = molecule['pos_in_seq'][molecule['idx'] == i]
            # peptide_pos_orderd = peptide_pos[peptide_idx-1] # pos starts at 1
            # (3) get graph name for elemnt in batch
            # print(molecule['graph_name'])
            if isinstance(molecule['graph_name'], str):
                graph_name = molecule['graph_name']
            else:
                graph_name = molecule['graph_name'][i]


            # samples will be overwritten because we have multiple samples (sampling BS=1 is okay)
            create_new_pdb(peptide_pos, peptide_idx, graph_name, run_id, time_step=time_step) # peptide_pos