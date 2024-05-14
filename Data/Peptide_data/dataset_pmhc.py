'''
load data into correct format:
1. just graphs with type features
2. graphs with additional energy labels
'''

import os
import h5py
import pickle
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

# with h5py.File('/gpfs/work3/0/einf2380/data/pMHCI/features_output_folder/GNN/residue/230201/residue-4162789.hdf5', 'r') as f5: print(f5.keys())
# datadir = '/gpfs/work3/0/einf2380/data/pMHCI/features_output_folder/GNN/residue/230201/'

class Peptide_MHC_Dataset(Dataset):
     
    def __init__(self, datadir, split='train', center=True, pickle_file=True):

        datadir_pickle = './Data/Peptide_data/pmhc_100K/'
        # datadir_pickle = datadir

        if pickle_file:

            datadir_pickle = './Data/Peptide_data/pmhc_100K/'
            # datadir_pickle = datadir

            with open(Path(datadir_pickle, 'dataset_pmhc.pkl'), 'rb') as f:
                self.data = pickle.load(f)

        else:

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
                'pos_in_seq': [],
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
                    chain_ids_protein_pocket = torch.tensor([1 if id == b'M' else 0 for id in chain_ids], dtype=torch.bool)
                    chain_ids_peptide = torch.tensor([0 if id == b'M' else 1 for id in chain_ids], dtype=torch.bool)

                    position = graph['node_features']['_position']
                    position = torch.tensor(position)
                    position_peptide = position[chain_ids_peptide]
                    position_protein_pocket = position[chain_ids_protein_pocket]

                    features = graph['node_features']['res_type']
                    features = torch.tensor(features)
                    features_peptide = features[chain_ids_peptide]
                    features_protein_pocket = features[chain_ids_protein_pocket]

                    feature_length = len(features_peptide) # num_nodes in each graph

                    # TODO: we should use these edges
                    edge_idx = graph['edge_features']['_index']
                    # whether edge is a covalent bound or not
                    edge_type = graph['edge_features']['covalent']

                    ## Adding positional AS_sequence information
                    node_names = graph['node_features']['_name']
                    pos_in_seq = torch.tensor([int(re.findall(r'\b(\d+)\b', str(node_name))[-1]) for node_name in node_names])
                    pos_in_seq_peptide = pos_in_seq[chain_ids_peptide]
                    if len(pos_in_seq_peptide) > torch.sum(chain_ids_peptide):
                        print(torch.sum(chain_ids_peptide))
                        raise ValueError

                    # pos_in_seq = torch.zeros(len(features_peptide)) - 1
                    # edge_idx = torch.tensor(edge_idx)
                    # edge_idx_covalent = edge_idx[torch.tensor(edge_type, dtype=torch.bool)]
                    # unique, counts = torch.unique(torch.cat((edge_idx_covalent[:,0], edge_idx_covalent[:,1])), return_counts=True)
                    # possible_starts = counts == 1
                    # start = unique[possible_starts][0]
                    # chain = [start.item()]
                    # current = start
                    # while len(chain) != len(unique):
                    #     next_aa = edge_idx_covalent[edge_idx_covalent[:,0] == current][:,1]
                    #     chain += [next_aa.item()]
                    #     current = next_aa
                    # for i, idx in enumerate(chain):
                    #     pos_in_seq[idx] = i

                    self.data['graph_name'].append([graph_name])

                    self.data['peptide_positions'].append(torch.tensor(position_peptide))
                    self.data['peptide_features'].append(torch.tensor(features_peptide))
                    self.data['num_peptide_residues'].append(feature_length)
                    self.data['peptide_idx'].append(torch.ones(len(position_peptide)))

                    self.data['protein_pocket_positions'].append(torch.tensor(position_protein_pocket))
                    self.data['protein_pocket_features'].append(torch.tensor(features_protein_pocket))
                    self.data['num_protein_pocket_residues'].append(feature_length)
                    self.data['protein_pocket_idx'].append(torch.ones(len(position_protein_pocket)))

                    self.data['pos_in_seq'].append(pos_in_seq_peptide)

            if center:
                for i in range(len(self.data['peptide_positions'])):
                    mean = (self.data['peptide_positions'][i].sum(0) +
                            self.data['protein_pocket_positions'][i].sum(0)) / \
                        (len(self.data['peptide_positions'][i]) + len(self.data['protein_pocket_positions'][i]))
                    self.data['peptide_positions'][i] = self.data['peptide_positions'][i] - mean
                    self.data['protein_pocket_positions'][i] = self.data['protein_pocket_positions'][i] - mean

            with open(Path(datadir_pickle, 'dataset_pmhc.pkl'), 'wb') as f:
                pickle.dump(self.data, f)

        
        # splitting dataset into train, val and test
        data_len = len(self.data['graph_name'])
        train_size = int(0.8 * data_len)
        val_size = int(0.1 * data_len)
        test_size = data_len - train_size - val_size
        train_set, val_set, test_set = {}, {}, {}
        for key in self.data.keys():
            train_set[key] = self.data[key][:train_size]
            val_set[key] = self.data[key][train_size:train_size+val_size]
            test_set[key] = self.data[key][train_size+val_size:]

        if split == 'train':
            self.dataset = train_set
        elif split == 'val':
            self.dataset = val_set
        else:
            self.dataset = test_set

    def __len__(self):
        return len(self.dataset['graph_name'])

    def __getitem__(self, idx):
        data = {key: val[idx] for key, val in self.dataset.items()}

        return data
    
    @staticmethod
    def collate_fn(batch):
        
        data_batch = {}
        for key in batch[0].keys():

            if key == 'graph_name':
                data_batch[key] = [x[key] for x in batch]
            elif key == 'num_peptide_residues' or key == 'num_protein_pocket_residues':
                data_batch[key] = torch.tensor([x[key] for x in batch])
            elif 'idx' in key:
                # make sure indices in batch start at zero (needed for torch_scatter)
                # This doesn't work as it should
                data_batch[key] = torch.cat([i * torch.ones(len(x[key])) for i, x in enumerate(batch)], dim=0)
            else:
                data_batch[key] = torch.cat([x[key] for x in batch], dim=0)

        return data_batch



