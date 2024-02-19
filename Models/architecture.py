import torch
import numpy as np
import torch.nn as nn

from Models.architectures.egnn import EGNN
from Models.architectures.gnn import GNN
from Models.architectures.ponita.models.ponita import Ponita

"""
This file sets up the neural network for the generative framework.
Before we can pass the data to our neural network we need to encode the molecule and the protein_pocket in a joint space.
"""

class NN_Model(nn.Module):

    def __init__(
            self,
            architecture: str,
            num_atoms: int,
            num_residues: int,
            joint_dim: int,
            hidden_dim: int,
            num_layers: int,

    ):
        # same encoder, decoders as in [Schneuing et al. 2023]
        self.atom_encoder = nn.Sequential(
            nn.Linear(num_atoms, 2 * num_atoms),
            nn.SiLU(),
            nn.Linear(2 * num_atoms, joint_dim)
        )

        self.atom_decoder = nn.Sequential(
            nn.Linear(joint_dim, 2 * num_atoms),
            nn.SiLU(),
            nn.Linear(2 * num_atoms, num_atoms)
        )

        self.residue_encoder = nn.Sequential(
            nn.Linear(num_residues, 2 * num_residues),
            nn.SiLU(),
            nn.Linear(2 * num_residues, joint_dim)
        )

        self.residue_decoder = nn.Sequential(
            nn.Linear(joint_dim, 2 * num_residues),
            nn.SiLU(),
            nn.Linear(2 * num_residues, num_residues)
        )

        # possible to use edge_embedding if I want to distinguish between molecule-moelcule, pocket-molecule edges, usw.

        # possible to condition on time (default is True)

        if architecture == 'ponita':

            # certain preprocessing steps necessary for ponita

            self.model = Ponita(in_channels_scalar + in_channels_vec,
                            hidden_dim,
                            out_channels_scalar,
                            num_layers,
                            output_dim_vec=out_channels_vec,
                            radius=args.radius,
                            num_ori=args.num_ori,
                            basis_dim=args.basis_dim,
                            degree=args.degree,
                            widening_factor=args.widening_factor,
                            layer_scale=args.layer_scale,
                            task_level='graph',
                            multiple_readouts=args.multiple_readouts,
                            lift_graph=True)
            
        elif architecture == 'egnn':

            self.model = EGNN()

        elif architecture == 'gnn':

            self.model = GNN()

        else:
            raise Exception(f"Wrong architecture {architecture}")





    def forward(self, z_t_mol, z_t_pro, t, molecule['idx'], protein_pocket['idx']):

        # encode z_t_mol
        # encode z_t_pro

        # run trough neural net model (# certain preprocessing steps necessary for ponita)

        # decode epsilon_hat_mol
        # decode epsilon_hat_pro

        return epsilon_hat_mol, epsilon_hat_pro