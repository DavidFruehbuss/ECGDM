import torch
import pytorch.lightning as pl

FLOAT_TYPE = torch.float32
INT_TYPE = torch.int64

from Models.diffusion_model import Conditional_Diffusion_Model
from Models.gflow_model import GFlow_Model

from Models.architecture import NN_Model

"""
This file implements the 3D-structure prediction for a moelcule 
and a protein with a conditional diffusion model.
"""

class Structure_Prediction_Model(pl.LightningModule):

    def __init__(
            self,
            dataset: str,
            generative_model: str,
            neural_network: str,
            network_params: dict,
            batch_size: int,
            lr: float,

    ):
        """
        Parameters:

        
        """
        
        super().__init__()

        # choose the generative framework
        frameworks = {'conditional_diffusion': Conditional_Diffusion_Model,
                     'generative_flow_network': GFlow_Model}
        assert generative_model in frameworks
        self.model = frameworks[generative_model]()

        # choose the neural net architecture
        self.neural_net = NN_Model(
            # model parameters
        )

        self.lr = lr

    # Data section

    def setup(self):
        raise NotImplementedError

    def train_dataloader():
        raise NotImplementedError

    def val_dataloader():
        raise NotImplementedError

    def test_dataloader():
        raise NotImplementedError
    
    def get_molecule_and_protein(self, data):
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
        return molecule, protein_pocket

    # training section

    def training_step(self, batch):
        loss = self.model(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch):
        loss = self.model(batch)
        self.log('val_loss', loss)

    def configure_optimizer(self):
        optimizer = torch.optim.AdamW(self.neural_net.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-12)
        return optimizer
    



    


