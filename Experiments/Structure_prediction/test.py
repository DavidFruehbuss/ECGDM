import argparse
from argparse import Namespace
from pathlib import Path
import yaml
import os
import time

import torch
import pytorch_lightning as pl
from torch_scatter import scatter_add
import wandb

from ECGDM.Experiments.Structure_prediction.lightning_module import Structure_Prediction_Model
from Data.Peptide_data.dataset_pmhc import Peptide_MHC_Dataset

if __name__ == "__main__":

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

    wandb.init(project=args.project, entity=args.entity, name=args.run_name,)

    num_samples = args.num_samples
    sample_batch_size = args.sample_batch_size

    lightning_model = Structure_Prediction_Model.load_from_checkpoint(
                    args.checkpoint,
                    dataset=args.dataset,
                    data_dir=args.data_dir,
                    dataset_params=args.dataset_params,
                    task_params=args.task_params,
                    generative_model=args.generative_model,
                    generative_model_params=args.generative_model_params,
                    architecture=args.architecture,
                    network_params=args.network_params,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    num_workers=args.num_workers,
                    device=args.device,
    )   

    lightning_model = lightning_model.to(device)
    lightning_model.setup('test')
    test_dataset = lightning_model.test_dataset

    results = []
    saved_samples = {}
    saved_samples['x_target'] = []
    saved_samples['x_predicted'] = []
    saved_samples['h'] = []

    for i in range(0, len(test_dataset), sample_batch_size):

        if i > 0: continue
        start_time = time.time()

        # prepare peptide-MHC
        mol_pro_list = [test_dataset[i+j] for _ in range(num_samples) for j in range(sample_batch_size)]
        mol_pro_samples = Peptide_MHC_Dataset.collate_fn(mol_pro_list)

        # sample new peptide-MHC structures using trained model
        mol_pro_batch = lightning_model.get_molecule_and_protein(mol_pro_samples)
        molecule, protein_pocket = mol_pro_batch
        xh_mol_final, xh_pro_final = lightning_model.model.sample_structure(num_samples, molecule, protein_pocket, wandb, i)

        # Safe resulting structures
        true_pos = [molecule['x'][molecule['idx']][i*num_samples] for i in range(sample_batch_size)]
        true_h = [molecule['h'][molecule['idx']][i*num_samples] for i in range(sample_batch_size)]
        saved_samples['x_target'] += true_pos[:]
        saved_samples['x_predicted'] += xh_mol_final[:,:3]
        saved_samples['h'] += true_h[:]
        print(saved_samples['x_target'][0].shape)
        print(saved_samples['x_predicted'][0].shape)
        print(saved_samples['h'][0].shape)

        # Calculate the RMSE error
        error_mol = scatter_add(torch.sqrt(torch.sum((molecule['x'] - xh_mol_final[:,:3])**2, dim=-1)), molecule['idx'], dim=0)
        # Normalize loss_t by graph size
        rmse = error_mol / ((3 + args.dataset_params.num_atoms) * molecule['size'])
        rmse_sample_mean = [rmse[j*num_samples:(j+1)*num_samples].mean(0) for j in range(sample_batch_size)]
        rmse_sample_best = [rmse[j*num_samples:(j+1)*num_samples].min(0) for j in range(sample_batch_size)]

        end_time = time.time()

        print([rmse, rmse_sample_mean, rmse_sample_best])
        print(f'Time: {end_time - start_time}')

