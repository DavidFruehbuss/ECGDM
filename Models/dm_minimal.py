import torch
import numpy as np
import math

import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean

from Models.noise_schedule import Noise_Schedule
from Data.Peptide_data.utils import create_new_pdb

class Conditional_Diffusion_Model(nn.Module):

    """
    Conditional Denoising Diffusion Model    
    """

    def __init__(
        self,
        neural_net: nn.Module,
        protein_pocket_fixed: bool,
        features_fixed: bool,
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

        # dataset info
        self.num_atoms = num_atoms
        self.num_residues = num_residues
        self.norm_values = norm_values
        self.x_dim = 3

        # Noise Schedule
        self.noise_schedule = Noise_Schedule(self.T)

        self.com_old = False
        
    def forward(self, z_data):

        molecule, protein_pocket = z_data
        molecule_pos = molecule['pos_in_seq']

        # compute noised sample
        z_t_mol, z_t_pro, epsilon_mol, epsilon_pro, t = self.noise_process(z_data)

        # use neural network to predict noise
        epsilon_hat_mol, epsilon_hat_pro = self.neural_net(z_t_mol, z_t_pro, t, molecule['idx'], protein_pocket['idx'], molecule_pos)

        # compute alpha, sigma
        alpha_t = self.noise_schedule(t, 'alpha')
        sigma_t = self.noise_schedule(t, 'sigma')

        # compute denoised sample
        z_data_hat = (1 / alpha_t)[molecule['idx']] * z_t_mol - (sigma_t / alpha_t)[molecule['idx']] * epsilon_hat_mol

        # Log sampling progress to wandb
        error_mol = scatter_add(torch.sum((molecule['x'] - z_data_hat[:,:3])**2, dim=-1), molecule['idx'], dim=0)
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

        # sample t ~ U(0,...,T) for each graph individually
        t_low = 0 if self.train else 1
        t = torch.randint(t_low, self.T + 1, size=(batch_size, 1), device=device)
        t = t / self.T

        # option for computing t = 0 representations
        t = torch.zeros((batch_size, 1), device=device) if t_is_0 else t
        
        # noise schedule
        alpha_t = self.noise_schedule(t, 'alpha')
        sigma_t = self.noise_schedule(t, 'sigma')

        # prepare joint point cloud
        xh_mol = torch.cat((molecule['x'], molecule['h']), dim=1)
        xh_pro = torch.cat((protein_pocket['x'], protein_pocket['h']), dim=1)

        # compute noise
        eps_x_mol = torch.randn(size=(len(xh_mol), self.x_dim), device=device)
        eps_x_pro = torch.zeros(size=(len(xh_pro), self.x_dim), device=device)

        eps_h_mol = torch.zeros(size=(len(xh_mol), self.num_atoms), device=device)
        eps_h_pro = torch.zeros(size=(len(xh_pro), self.num_residues), device=device)

        epsilon_mol = torch.cat((eps_x_mol, eps_h_mol), dim=1)
        epsilon_pro = torch.cat((eps_x_pro, eps_h_pro), dim=1)

        # compute noised representations
        z_t_mol = alpha_t[molecule['idx']] * xh_mol + sigma_t[molecule['idx']] * epsilon_mol
        z_t_pro = xh_pro

        return z_t_mol, z_t_pro, epsilon_mol, epsilon_pro, t
    
    def train_loss(
            self, molecule, z_t_mol, epsilon_mol, epsilon_hat_mol,
            protein_pocket, z_t_pro, epsilon_pro, epsilon_hat_pro, 
            t,
    ):
        # compute the sum squared error loss per graph
        error_mol = scatter_add(torch.sum((epsilon_mol - epsilon_hat_mol)**2, dim=-1), molecule['idx'], dim=0)
        error_pro = torch.zeros(protein_pocket['size'].size(0), device=molecule['x'].device)
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
        molecule_pos = molecule['pos_in_seq']

        # compute the sum squared error loss per graph
        error_mol = scatter_add(torch.sum((epsilon_mol - epsilon_hat_mol)**2, dim=-1), molecule['idx'], dim=0)

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
        ones = torch.ones_like(sigma_T_x)
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
        molecule_pos = molecule['pos_in_seq']
        num_samples = len(molecule['size'])

        # Record protein_pocket center of mass before
        protein_pocket_com_before = scatter_mean(protein_pocket['x'], protein_pocket['idx'], dim=0)

        # Presteps

        # If we want a new peptide we would have to add random sampling of the start peptide here^

        # define the target at the pocket position
        mol_target_p = molecule['x']
        mol_target_p = mol_target_p + protein_pocket_com_before[molecule['idx']]

        # define the target at 0-COM
        mol_target_0 = molecule['x']

        # Normalisation
        # molecule['x'] = molecule['x'] / self.norm_values[0]
        molecule['h'] = molecule['h'] / self.norm_values[1]
        protein_pocket['x'] = protein_pocket['x'] / self.norm_values[0]
        protein_pocket['h'] = protein_pocket['h'] / self.norm_values[1]

        # start with random peptide position (target hidden)
        # mean=COM, sigma=1, and sample epsioln
        rand_eps_x = torch.randn((len(molecule['x']), self.x_dim), device=device)
        molecule_x = protein_pocket_com_before[molecule['idx']] + rand_eps_x

        # TODO: here features get changed! (for peptides need to turn this off somehow)
        if self.features_fixed:
            molecule_h = molecule['h']
        else:
            # for this starting h and updating would change
            print('Shouldnt be reached')
            molecule_h = torch.zeros((protein_pocket['h'].size), device=device)

        # combine position and features
        xh_mol = torch.cat((molecule_x, molecule_h), dim=1)
        xh_pro = torch.cat((protein_pocket['x'], protein_pocket['h']), dim=1)

        error_mol = scatter_add(torch.sum((mol_target_p - xh_mol[:,:3])**2, dim=-1), molecule['idx'], dim=0)
        rmse = torch.sqrt(error_mol / (3 * molecule['size']))
        print(f'The starting RSME of random noise is {rmse.mean(0)}')

        error_mol = scatter_add(torch.sum((mol_target_0 - xh_mol[:,:3])**2, dim=-1), molecule['idx'], dim=0)
        rmse = torch.sqrt(error_mol / (3 * molecule['size']))
        print(f'The starting RSME of random noise at COM 0 is {rmse.mean(0)}')

        # # Sanaty check
        molecule_xx = molecule['x']
        xh_t_mol = torch.cat((molecule_xx, molecule_h), dim=1)
        z_t_pro = xh_pro

        error_mol = scatter_add(torch.sum((mol_target_0 - molecule_xx[:,:3])**2, dim=-1), molecule['idx'], dim=0)
        rmse = torch.sqrt(error_mol / (3 * molecule['size']))
        print(f'Sanity check 1 (self): {rmse.mean(0)}')

        eps_x_mol = torch.randn(size=(len(xh_mol), self.x_dim), device=device)

        if self.features_fixed:
            eps_h_mol = torch.zeros(size=(len(xh_mol), self.num_atoms), device=device)

        epsilon_mol = torch.cat((eps_x_mol, eps_h_mol), dim=1)

        if self.T == 1000:
            TS = [1, 50, 100, 200, 400, 500, 700, 800, 900, 950, 998]
        else:
            TS = [1, 2, 10, 25, 50, 100, 200, 300, 400, 450, 498]

        for ts in TS:

            t_array = torch.randint(ts, ts + 1, size=(num_samples, 1), device=device)
            # normalize t
            t_array = t_array / self.T
            alpha_t = self.noise_schedule(t_array, 'alpha')
            sigma_t = self.noise_schedule(t_array, 'sigma')

            ## define very low sigma, alpha
            # sigma_t = torch.tensor([0.0001]).to(molecule['x'].device)
            # alpha_t = torch.sqrt(1-sigma_t**2)

            self.safe_pdbs(xh_t_mol, molecule, run_id, time_step=f'start')

            z_t_mol = alpha_t[molecule['idx']] * xh_t_mol + sigma_t[molecule['idx']] * epsilon_mol

            self.safe_pdbs(z_t_mol, molecule, run_id, time_step=f'N_{ts}')

            # use neural network to predict noise
            epsilon_hat_mol, epsilon_hat_pro = self.neural_net(z_t_mol, z_t_pro, t_array, molecule['idx'], protein_pocket['idx'], molecule_pos)

            # compute denoised sample
            # original equation
            z_data_hat = (1 / alpha_t)[molecule['idx']] * z_t_mol - (sigma_t / alpha_t)[molecule['idx']] * epsilon_hat_mol
            z_data_hat_2 = (1 / alpha_t)[molecule['idx']] * z_t_mol - (sigma_t / alpha_t)[molecule['idx']] * epsilon_mol

            # expanend_fraction = (sigma_t / alpha_t)[molecule['idx']]
            # print(f'sigma {sigma_t.shape}')
            # print(f'alpha {alpha_t.shape}')
            # print(f'sigma/alpha {(sigma_t / alpha_t).shape}')
            # print(f'alpha[mol_idx] {expanend_fraction.shape}')

            error_mol = scatter_add(torch.sum((molecule_xx - z_data_hat[:,:3])**2, dim=-1), molecule['idx'], dim=0)
            rmse = torch.sqrt(error_mol / (3 * molecule['size']))
            print(f'Sanity Check 2 (normal) {rmse.mean(0)}')

            error_mol = scatter_add(torch.sum((molecule_xx - z_data_hat_2[:,:3])**2, dim=-1), molecule['idx'], dim=0)
            rmse = torch.sqrt(error_mol / (3 * molecule['size']))
            print(f'Sanity Check 3 (true noise) {rmse.mean(0)}')

            self.safe_pdbs(z_data_hat, molecule, run_id, time_step=f'DN{ts}')

        # Visualize true peptide pmhc
        self.safe_pdbs(z_data_hat_2, molecule, run_id, time_step=f'T{ts}')

        ##################
        # Sanity Check with 500 steps from 500 noised sample
        # xh_mol = z_t_mol
        ##################


        # Iterativly denoise stepwise for t = T,...,1
        for s in reversed(range(0,self.T)):

            if s % 100 == 0:
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
            eps_mol_random = torch.randn(size=(len(xh_mol), self.x_dim + self.num_atoms), device=device)
            eps_mol_random = eps_mol_random - scatter_mean(eps_mol_random, molecule['idx'], dim=0)[molecule['idx']]
            # the line bellow is where we would add the gaudi guidance (-> compute backbone and statistical potentials)
            xh_mol = mean_mol_s + sigma_mol_s[molecule['idx']] * eps_mol_random
            xh_pro = xh_pro.detach().clone() # for safety (probally not necessary)

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
        eps_lig_random = torch.randn(size=(len(xh_mol), self.x_dim + self.num_atoms), device=device)
        xh_mol_final = mean_mol_final + sigma_mol_final[molecule['idx']] * eps_lig_random
        xh_pro_final = xh_pro.detach().clone() # for safety (probally not necessary)

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
        mol_target = molecule['x']

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

        for i in range(len(molecule['size'])):
            # (1) extract the peptide position
            pos = pos[:,:3]
            peptide_pos = pos[molecule['idx'] == i]
            # (2) bring peptides into correct order
            peptide_idx = molecule['pos_in_seq'][molecule['idx'] == i]
            peptide_pos_orderd = peptide_pos[peptide_idx-1] # pos starts at 1
            # orderings are different to base ordering
            # but they are also different to the pdb ordering
            # (3) get graph name for elemnt in batch
            graph_name = molecule['graph_name'][i]

            # samples will be overwritten because we have multiple samples (sampling BS=1 is okay)
            create_new_pdb(peptide_pos_orderd, graph_name[0], run_id, time_step=time_step)