'''
load data into correct format:
1. just graphs with type features
2. graphs with additional energy labels
'''

import os
import h5py

import numpy as np
import torch
from torch.utils.data import Dataset

# with h5py.File('/gpfs/work3/0/einf2380/data/pMHCI/features_output_folder/GNN/residue/230201/residue-4162789.hdf5', 'r') as f5: print(f5.keys())
# datadir = '/gpfs/work3/0/einf2380/data/pMHCI/features_output_folder/GNN/residue/230201/'

class Peptide_MHC_Dataset(Dataset):
     
    def __init__(self, datadir, center=True):

        self.data = {
            'graph_name': [],
            'peptide_positions': [],
            'peptide_features': [],
            'num_peptide_residues': [],
            'peptide_idx': [],
            'protein_pocket_positions': [],
            'protein_pocket_features': [],
            'num_protein_pocket_residues': [],
            'protein_pocket_idx': [],
            'edge_idx': [],
            'edge_type': [],
        }

        for filename in os.listdir(datadir):
                
            # need if statment to only select h5py files
            if not filename.endswith('.hdf5'):
                continue
            
            file_path = os.path.join(datadir, filename)

            data_subset = h5py.File(file_path, 'r')

            for graph_name, graph in data_subset.items():

                # get node ids for peptide (0) and protein (1)
                chain_ids = graph['node_features']['_chain_id']
                chain_ids_protein_pocket = [1 if id == b'M' else 0 for id in chain_ids]
                chain_ids_peptide = [0 if id == b'M' else 1 for id in chain_ids]

                position = graph['node_features']['_position']
                position = torch.tensor(position)
                position_peptide = position[chain_ids_peptide]
                position_protein_pocket = position[chain_ids_protein_pocket]

                features = graph['node_features']['res_type']
                features = torch.tensor(features)
                features_peptide = features[chain_ids_peptide]
                features_protein_pocket = features[chain_ids_protein_pocket]

                feature_length = len(features_peptide[0]) # always 20 because we only have residues

                # TODO: we should use these edges
                edge_idx = graph['edge_features']['_index']
                # whether edge is a covalent bound or not
                edge_type = graph['edge_features']['covalent']

                self.data['graph_name'].append([graph_name])

                self.data['peptide_positions'].append([position_peptide])
                self.data['peptide_features'].append([features_peptide])
                self.data['num_peptide_residues'].append([feature_length])
                self.data['peptide_idx'].append([torch.ones(len(position_peptide))])

                self.data['protein_pocket_positions'].append([position_protein_pocket])
                self.data['protein_pocket_features'].append([features_protein_pocket])
                self.data['num_protein_pocket_residues'].append([feature_length])
                self.data['protein_pocket_idx'].append([torch.ones(len(position_protein_pocket))])

        if center:
            for i in range(len(self.data['peptide_positions'])):
                mean = (self.data['peptide_positions'][i].sum(0) +
                        self.data['protein_pocket_positions'][i].sum(0)) / \
                       (len(self.data['peptide_positions'][i]) + len(self.data['protein_pocket_positions'][i]))
                self.data['peptide_positions'][i] = self.data['peptide_positions'][i] - mean
                self.data['protein_pocket_positions'][i] = self.data['protein_pocket_positions'][i] - mean

    def __len__(self):
        return len(self.data['graph_name'])

    def __getitem__(self, idx):
        data = {key: val[idx] for key, val in self.data.items()}

        return data
    
    def collate_fn(batch):
        
        data_batch = {}
        for key in batch[0].keys():

            if key == 'graph_name':
                data_batch[key] = [x[key] for x in batch]
            elif key == 'num_peptide_residues' or key == 'num_protein_pocket_residues':
                data_batch[key] = torch.tensor([x[key] for x in batch])
            elif 'idx' in key:
                # make sure indices in batch start at zero (needed for torch_scatter)
                data_batch[key] = torch.cat([i * torch.ones(len(x[key])) for i, x in enumerate(batch)], dim=0)
            else:
                data_batch[key] = torch.cat([x[key] for x in batch], dim=0)

        return data_batch



