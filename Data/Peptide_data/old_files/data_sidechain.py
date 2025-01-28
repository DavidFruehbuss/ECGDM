# File modified from https://github.com/cbaakman/diffusion-model/

import os
import h5py
import re
from typing import Dict
from typing import Optional

import torch
from torch.utils.data import Dataset

from Data.Peptide_data.openfold_utils import Rigid


class Backbone_Dataset(Dataset):
     
    def __init__(self, datadir, split='train_valid'):

        self.hdf5_path = f'{datadir}100k_{split}.hdf5'

        with h5py.File(self.hdf5_path, 'r') as f5:
            self.entry_names = list(f5.keys())

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.get_entry(self.entry_names[index])

    def __len__(self) -> int:
        return len(self.entry_names)
    
    def get_entry(self, entry_name: str) -> Dict[str, torch.Tensor]:

        data = {}
        with h5py.File(self.hdf5_path, 'r') as f5:
            entry = f5[entry_name]

            if "peptide" not in entry:
                raise ValueError(f"no peptide in {entry_name}")

            peptide = entry["peptide"]
            mhc = entry['protein']

            # backbone rotation(quaternion) + c-alpha xyz
            frames_data = torch.tensor(peptide['backbone_rigid_tensor'][:]) # N * 4 * 4
            mhc_frames_data = torch.tensor(mhc['backbone_rigid_tensor'][:]) # N * 4 * 4

            peptide_7 = Rigid.from_tensor_4x4(frames_data).to_tensor_7()
            mhc_7 = Rigid.from_tensor_4x4(mhc_frames_data).to_tensor_7()

            sidechain_train = torch.cat((peptide_7[:,4:], mhc_7[:,4:]), dim=0) # M * 3
            sidechain_q4_labels = peptide_7[:,:4] # N * 4

            peptide_len = frames_data.shape[0]
            mhc_len = mhc_frames_data.shape[0]

            # backbone reswise mask
            mask = torch.ones(peptide_len, dtype=torch.bool)
            mhc_mask = torch.ones(mhc_len, dtype=torch.bool)

            # one-hot encoded amino acid sequence
            onehot = torch.tensor(peptide['sequence_onehot'][:,:20])
            mhc_onehot = torch.tensor(mhc['sequence_onehot'][:,:20])

            # output dict
            data['graph_name'] = entry_name
            data['peptide_positions'] = peptide_7[:,4:]
            data['peptide_idx'] = mask
            data['peptide_features'] = onehot
            data['num_peptide_residues'] = peptide_len
            data['protein_pocket_idx'] = mhc_mask
            data['protein_pocket_positions'] = mhc_7[:,4:]
            data['protein_pocket_features'] = mhc_onehot
            data['num_protein_pocket_residues'] = mhc_len
            data['sidechain_train'] = sidechain_train
            data['sidechain_labels'] = sidechain_q4_labels

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



