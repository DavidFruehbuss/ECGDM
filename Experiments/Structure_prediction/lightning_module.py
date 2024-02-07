import torch
import pytorch.lightning as pl

from Models.diffusion_model import Conditional_Diffusion_Model
from Models.gflow_model import GFlow_Model

from Models.architectures import ponita, egnn, gnn 

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
            num_epochs: int,

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
        neural_networks = {'PONITA': ponita,
                          'EGNN': egnn,
                          'GNN': gnn}
        self.neural_net = neural_networks[neural_network]()



    def training_step(self, batch):
        loss = self.model(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch):
        loss = self.model(batch)
        self.log('val_loss', loss)

    def configure_optimizer(self):
        optimizer = 
        return optimizer


    


