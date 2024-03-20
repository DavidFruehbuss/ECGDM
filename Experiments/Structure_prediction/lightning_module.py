import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pathlib import Path

FLOAT_TYPE = torch.float32
INT_TYPE = torch.int64

from Data.Ligand_data.dataset_ligand import ProcessedLigandPocketDataset
from Data.Peptide_data.dataset_pmhc import Peptide_MHC_Dataset

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
            task_params: dict,
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

        # set a seed
        torch.manual_seed(42)

        # choose the generative framework
        frameworks = {'conditional_diffusion': Conditional_Diffusion_Model} # , 'generative_flow_network': GFlow_Model
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
            task_params.features_fixed,
            generative_model_params.timesteps,
            dataset_params.num_atoms,
            dataset_params.num_residues,
            dataset_params.norm_values,
        )
        
        self.dataset = dataset
        self.data_dir = data_dir
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers

        # transform for ligand data
        if self.dataset == 'ligand':
            self.data_transform = None

    # Data section

    def setup(self, stage):
        if self.dataset == 'pmhc':

            dataset = Peptide_MHC_Dataset(self.data_dir)
            # split into training, validation and test data
            train_size = int(0.8 * len(dataset))
            val_size = int(0.1 * len(dataset))
            test_size = len(dataset) - train_size - val_size
            train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
            if stage == 'fit':
                self.train_dataset = train_set
                self.val_dataset = val_set
            elif stage == 'test':
                self.test_dataset = test_set

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
        if self.dataset == 'pmhc':

            molecule = {
                'x': data['peptide_positions'].to(self.device, FLOAT_TYPE),
                'h': data['peptide_features'].to(self.device, FLOAT_TYPE),
                'size': data['num_peptide_residues'].to(self.device, INT_TYPE),
                'idx': data['peptide_idx'].to(self.device, INT_TYPE),
            }

            protein_pocket = {
                'x': data['protein_pocket_positions'].to(self.device, FLOAT_TYPE),
                'h': data['protein_pocket_features'].to(self.device, FLOAT_TYPE),
                'size': data['num_protein_pocket_residues'].to(self.device, INT_TYPE),
                'idx': data['protein_pocket_idx'].to(self.device, INT_TYPE)
            }

        elif self.dataset == 'ligand':
        
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

        else:
            raise Exception(f"Wrong dataset {self.dataset}")
        

    # training section

    def training_step(self, data_batch):
        mol_pro_batch = self.get_molecule_and_protein(data_batch)
        # TODO: could add augment_noise and augment_rotation but excluded in DiffDock
        loss, info = self.model(mol_pro_batch)
        self.log('train_loss', loss)

        for key, value in info.items():
            self.log(key, value)

        return loss

    def validation_step(self, data_batch, *args):
        mol_pro_batch = self.get_molecule_and_protein(data_batch)
        loss, info = self.model(mol_pro_batch)
        self.log('val_loss', loss)

        # for key, value in info.items():
        #     val_key = key + 'val'
        #     self.log(val_key, value)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.neural_net.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-12)
        return optimizer
    



    


