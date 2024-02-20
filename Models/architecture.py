import torch
import numpy as np
import torch.nn as nn
from torch_scatter import scatter_mean

from Models.architectures.egnn import EGNN, GNN
from Models.architectures.ponita.models.ponita import Ponita

"""
This file sets up the neural network for the generative framework.
Before we can pass the data to our neural network we need to encode the molecule and the protein_pocket in a joint space.
"""

class NN_Model(nn.Module):

    def __init__(
            self,
            architecture: str,
            network_params,
            num_atoms: int,
            num_residues: int,
            joint_dim: int,
            hidden_dim: int,
            num_layers: int,
            pocket_position_fixed: bool,
            # device, TODO: add device handling

    ):
        self.architecture = architecture
        self.x_dim = 3
        self.pocket_position_fixed = pocket_position_fixed

        # possible to use edge_embedding if I want to distinguish between molecule-moelcule, pocket-molecule edges, usw.

        # possible to condition on time (default is True)

        if architecture == 'ponita':

            self.atom_encoder = nn.Linear(num_atoms, joint_dim)

            self.atom_decoder = nn.Linear(joint_dim, num_atoms)

            self.residue_encoder = nn.Linear(num_residues, joint_dim)

            self.residue_decoder = nn.Linear(joint_dim, num_residues)

            # dimensions for ponita model
            in_channels_scalar = joint_dim # + 1 for time
            in_channels_vec = 0
            # really unsure what to use here
            out_channels_scalar = joint_dim # updated features
            out_channels_vec = 1 # displacment vector

            self.ponita = Ponita(in_channels_scalar + in_channels_vec,
                            hidden_dim,
                            out_channels_scalar,
                            num_layers,
                            output_dim_vec=out_channels_vec,
                            radius=network_params.radius,
                            num_ori=network_params.num_ori,
                            basis_dim=network_params.basis_dim,
                            degree=network_params.degree,
                            widening_factor=network_params.widening_factor,
                            layer_scale=network_params.layer_scale,
                            task_level='graph',
                            multiple_readouts=network_params.multiple_readouts,
                            lift_graph=True)
            
        elif architecture == 'egnn' or 'gnn':

            # same encoder, decoders as in [Schneuing et al. 2023]
            self.atom_encoder = nn.Sequential(
                nn.Linear(num_atoms, 2 * num_atoms),
                nn.SiLU(),
                nn.inear(2 * num_atoms, joint_dim)
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

            if architecture == 'egnn':

                # TODO: initialize EGNN
                self.egnn = EGNN()

            else:
                
                # TODO: initialize GNN
                self.gnn = GNN()

        else:
            raise Exception(f"Wrong architecture {architecture}")





    def forward(self, z_t_mol, z_t_pro, t, molecule_idx, protein_pocket_idx):

        '''
        Inputs:
        z_t_mol: [batch_node_dim_mol, x + num_atoms]
        z_t_pro: [batch_node_dim_pro, x + num_residues]
        t: int
        molecule['idx']: [batch_node_dim]
        protein_pocket['idx']: [batch_node_dim]

        return epsilon_hat_mol [batch_node_dim_mol, x + num_atoms], 
                epsilon_hat_pro [batch_node_dim_pro, x + num_residues]
        '''

        # add edges to the graph
        # TODO: add function for generating edges
        edges = None

        if self.architecture == 'ponita':
             
            # TODO: implement step 1-6 for ponita

            # (1) need z_t_mol and z_t_pro to be of the same size but no nolinear embedding

            # (2) need to save [x, h, edges] as [graph.pos, graph.x, graph.edge_index]
            graph = None

            # (3) add time conditioning

            # (4) choose whether to get protein_pocket corrdinates fixed (might need to modify ponita)
            pocket_position_fixed = None if self.pocket_position_fixed \
                else torch.cat((torch.ones_like(molecule_idx), torch.ones_like(protein_pocket_idx))).unsqueeze(1)

            # (5) ponita forward pass (x_new could also be the displacment vector directly)
            h_new, x_new = self.ponita(graph)

            # (6) calculate displacement vectors (possibly not necessary see step 5.)
            displacement_vec = (x_new - x_joint)

            
        elif self.architecture == 'egnn' or self.architecture == 'gnn':

            # encode z_t_mol, z_t_pro (possible need to .clone() the inputs)
            h_mol = self.atom_encoder(z_t_mol[:,self.x_dim:])
            h_pro = self.residue_encoder(z_t_pro[:,self.x_dim:])

            # combine molecule and protein in joint space
            x_joint = torch.cat(z_t_mol[:,:self.x_dim], z_t_pro[:,:self.x_dim], dim=0) # [batch_node_dim_mol + batch_node_dim_pro, 3]
            h_joint = torch.cat(h_mol, h_pro, dim=0) # [batch_node_dim_mol + batch_node_dim_pro, joint_dim]
            idx_joint = torch.cat(molecule_idx, protein_pocket_idx, dim=0)

            # TODO: add time conditioning (1)

            # TODO: add edge embedding
            edge_types = None

            if self.architecture == 'egnn':

                # choose whether to get protein_pocket corrdinates fixed
                pocket_position_fixed = None if self.pocket_position_fixed \
                    else torch.cat((torch.ones_like(molecule_idx), torch.ones_like(protein_pocket_idx))).unsqueeze(1)

                # neural net forward pass
                h_new, x_new = self.egnn(h_joint, x_joint, edges,
                                            update_coords_mask=pocket_position_fixed,
                                            batch_mask=idx_joint, edge_attr=edge_types)
                
                # calculate displacement vectors
                displacement_vec = (x_new - x_joint)

            else:

                # GNN
                x_h_joint = torch.cat([x_joint, h_joint], dim=1)
                out = self.gnn(x_h_joint, edges, node_mask=None, edge_attr=edge_types)
                displacement_vec = out[:, :self.x_dim]
                h_new = out[:, self.x_dim:]


        else:
            raise Exception(f"Wrong architecture {self.architecture}")
        
        # TODO: add time conditioning (2)
                
        # decode h_new
        h_new_mol = self.atom_decoder(h_new[:len(molecule_idx)])
        h_new_pro = self.residue_decoder(h_new[len(molecule_idx):])

        # might not be necessary but let's see
        if torch.any(torch.isnan(displacement_vec)):
            raise ValueError("NaN detected in EGNN output")
        
        # remove mean batch of the position (not sure why)
        displacement_vec = displacement_vec - scatter_mean(displacement_vec, idx_joint, dim=0)

        # output
        epsilon_hat_mol = torch.cat(displacement_vec[:len(molecule_idx)], h_new_mol, dim=1)
        epsilon_hat_pro = torch.cat(displacement_vec[len(molecule_idx):], h_new_pro, dim=1)

        return epsilon_hat_mol, epsilon_hat_pro