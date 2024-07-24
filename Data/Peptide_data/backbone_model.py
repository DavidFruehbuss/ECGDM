import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

import sys
import os

def get_edges(batch_mask_ligand, batch_mask_pocket, x_ligand, x_pocket): 
        
        edge_cutoff_l = None
        edge_cutoff_i = 14.0
        edge_cutoff_p = 8.0

        adj_ligand = batch_mask_ligand[:, None] == batch_mask_ligand[None, :]
        adj_pocket = batch_mask_pocket[:, None] == batch_mask_pocket[None, :]
        adj_cross = batch_mask_ligand[:, None] == batch_mask_pocket[None, :]

        if edge_cutoff_l is not None:
            adj_ligand = adj_ligand & (torch.cdist(x_ligand, x_ligand) <= edge_cutoff_l)

        if edge_cutoff_p is not None:
            adj_pocket = adj_pocket & (torch.cdist(x_pocket, x_pocket) <= edge_cutoff_p)

        if edge_cutoff_i is not None:
            adj_cross = adj_cross & (torch.cdist(x_ligand, x_pocket) <= edge_cutoff_i)

        adj = torch.cat((torch.cat((adj_ligand, adj_cross), dim=1),
                         torch.cat((adj_cross.T, adj_pocket), dim=1)), dim=0)
        edges = torch.stack(torch.where(adj), dim=0)

        return edges
    
if __name__ == '__main__':

    desired_directory = '/gpfs/home4/dfruhbuss/ECGDM/'
    os.chdir(desired_directory)
    sys.path.insert(0, desired_directory)
    from Models.architectures.egnn import GNN
    from Data.Peptide_data.data_sidechain import Backbone_Dataset

    epochs = 100
    batch_size = 64
    data_dir = '/gpfs/work3/0/einf2380/data/tcrspec/'


    train_dataset = Backbone_Dataset(data_dir, 'train')
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
                          collate_fn=train_dataset.collate_fn,
                          pin_memory=True)
    
    gnn = GNN(in_node_nf=23, in_edge_nf=8,
                hidden_nf=64, out_node_nf=4,
                device='cuda', act_fn=nn.SiLU(), n_layers=3,
                attention=True, normalization_factor=100,
                aggregation_method='sum')
    edge_embedding = nn.Embedding(3, 8, device='cuda')
    
    optimizer = Adam(gnn.parameters(), lr = 1e-3)
    criterion = nn.MSELoss()
    
    for e in range(epochs):

        epoch_loss = 0

        for i, batch in enumerate(train_loader):

            print(batch['peptide_positions'].shape)
            print(batch['protein_pocket_positions'].shape)

            pos = torch.cat((batch['peptide_positions'], batch['protein_pocket_positions']),dim=0)
            features = torch.cat((batch['peptide_features'], batch['protein_pocket_features']),dim=0)
            print(pos.shape)
            print(features.shape)
            input = torch.cat((pos, features), dim=1)
            targets = batch['sidechain_labels']

            molecule_idx = batch['peptide_idx'].to('cuda')
            protein_pocket_idx = batch['protein_pocket_idx'].to('cuda')
            x_mol = batch['peptide_positions'].to('cuda')
            x_pro = batch['protein_pocket_positions'].to('cuda')
            edges = get_edges(molecule_idx, protein_pocket_idx, x_mol, x_pro)

            # 0: ligand-pocket, 1: ligand-ligand, 2: pocket-pocket
            edge_types = torch.zeros(edges.size(1), dtype=int, device=edges.device)
            edge_types[(edges[0] < len(molecule_idx)) & (edges[1] < len(molecule_idx))] = 1
            edge_types[(edges[0] >= len(molecule_idx)) & (edges[1] >= len(molecule_idx))] = 2

            # Learnable embedding
            edge_types = edge_embedding(edge_types)
            
            input = input.to('cuda')
            targets = targets.to('cuda')
            output = gnn(input, edges, node_mask=None, edge_attr=None)
            loss = criterion(output,targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss

    print(epoch_loss)

    



        
    
    