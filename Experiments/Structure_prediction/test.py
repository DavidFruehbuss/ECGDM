import argparse
from argparse import Namespace
from pathlib import Path
import yaml

import torch
import pytorch_lightning as pl
from torch_scatter import scatter_add

from ECGDM.Experiments.Structure_prediction.lightning_module import Structure_Prediction_Model
from Data.Peptide_data.dataset_pmhc import Peptide_MHC_Dataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# read in config
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

args_dict = args.__dict__
for key, value in config.items():
    if isinstance(value, dict):
        args_dict[key] = Namespace(**value)
    else:
        args_dict[key] = value

num_samples = args.num_samples

lightning_model = Structure_Prediction_Model.load_from_checkpoint(args.checkpoint)
lightning_model = lightning_model.to(device)

test_dataset = lightning_model.test_dataset

results = []

for i, mol_pro in enumerate(test_dataset):

    # prepare peptide-MHC
    molecule, protein_pocket = mol_pro
    mol_pro_list = [(mol_pro) for i in range(num_samples)]
    mol_pro_samples = Peptide_MHC_Dataset.collate_fn(mol_pro_list)

    # sample new peptide-MHC structures using trained model
    mol_pro_batch = lightning_model.get_molecule_and_protein(mol_pro_samples)
    xh_mol_final, xh_pro_final = lightning_model.model.sample(mol_pro_batch)

    # Calculate the RMSE error
    rmse = scatter_add(torch.sqrt(torch.sum((molecule['x'] - xh_mol_final[:,:3])**2, dim=-1)), molecule['idx'], dim=0)

    results += [mol_pro, (xh_mol_final, xh_pro_final), rmse]

print(results)

