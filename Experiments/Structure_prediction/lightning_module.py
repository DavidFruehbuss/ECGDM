import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pathlib import Path

FLOAT_TYPE = torch.float32
INT_TYPE = torch.int64

from Data.Ligand_data.dataset_ligand import ProcessedLigandPocketDataset

from Models.diffusion_model import Conditional_Diffusion_Model
# from Models.gflow_model import GFlow_Model

from Models.architecture import NN_Model

"""
This file implements the 3D-structure prediction for a moelcule 
and a protein with a conditional diffusion model.
"""

class Structure_Prediction_Model(pl.LightningModule):

    def __init__(
            self,
            dataset: str,
            data_dir: str,
            dataset_params: dict,
            generative_model: str,
            generative_model_params: dict,
            architecture: str,
            network_params: dict,
            batch_size: int,
            lr: float,
            num_workers: int,
            device,

    ):
        """
        Parameters:

        
        """
        
        super().__init__()

        # choose the generative framework
        frameworks = {'conditional_diffusion': Conditional_Diffusion_Model,
         #            'generative_flow_network': GFlow_Model
                     }
        assert generative_model in frameworks

        # choose the neural net architecture
        self.neural_net = NN_Model(
            # model parameters
            architecture,
            network_params,
            dataset_params.num_atoms,
            dataset_params.num_residues,
            device,
        )

        self.model = frameworks[generative_model](
            # framework parameters
            self.neural_net,
            generative_model_params.timesteps,
            dataset_params.num_atoms,
            dataset_params.num_residues,
        )
        
        self.dataset = dataset
        self.datadir = data_dir
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers

    # Data section

    def setup(self, stage):
        if self.dataset == 'pmhc':

            raise NotImplementedError

        elif self.dataset == 'ligand':
            if stage == 'fit':
                self.train_dataset = ProcessedLigandPocketDataset(
                    Path(self.datadir, 'train.npz'), transform=self.data_transform)
                self.val_dataset = ProcessedLigandPocketDataset(
                    Path(self.datadir, 'val.npz'), transform=self.data_transform)
            elif stage == 'test':
                self.test_dataset = ProcessedLigandPocketDataset(
                    Path(self.datadir, 'test.npz'), transform=self.data_transform)
            else:
                raise NotImplementedError
            
        else:
            raise Exception(f"Wrong dataset {self.dataset}")

    def train_dataloader(self):
        # Need to pick a dataloader (geometric or normal)
        # what does pin_memory do?
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=self.train_dataset.collate_fn,
                          pin_memory=True)
        raise NotImplementedError

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.val_dataset.collate_fn,
                          pin_memory=True)
        raise NotImplementedError

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.test_dataset.collate_fn,
                          pin_memory=True)
        raise NotImplementedError
    
    def get_molecule_and_protein(self, data):
        '''
        function to unpack the molecule and it's protein
        '''
        molecule = {
            'x': data['lig_coords'].to(self.device, FLOAT_TYPE),
            'h': data['lig_one_hot'].to(self.device, FLOAT_TYPE),
            'size': data['num_lig_atoms'].to(self.device, INT_TYPE),
            'idx': data['lig_mask'].to(self.device, INT_TYPE),
        }

        protein_pocket = {
            'x': data['pocket_coords'].to(self.device, FLOAT_TYPE),
            'h': data['pocket_one_hot'].to(self.device, FLOAT_TYPE),
            'size': data['num_pocket_nodes'].to(self.device, INT_TYPE),
            'idx': data['pocket_mask'].to(self.device, INT_TYPE)
        }
        return (molecule, protein_pocket)

    # training section

    def training_step(self, data_batch):
        mol_pro_batch = self.get_molecule_and_protein(data_batch)
        loss = self.model(mol_pro_batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, data_batch):
        mol_pro_batch = self.get_molecule_and_protein(data_batch)
        loss = self.model(mol_pro_batch)
        self.log('val_loss', loss)

    def configure_optimizer(self):
        optimizer = torch.optim.AdamW(self.neural_net.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-12)
        return optimizer
    



    


