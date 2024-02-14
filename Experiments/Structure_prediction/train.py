import argparse
from argparse import Namespace
import yaml

import pytorch_lightning as pl
from lightning_modules import Structure_Prediction_Model

if __name__ == "__main__":

    # read in config
    parser = argparse.ArgumentParser()
    parser.argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.save_load(f)

    args_dict = args.__dict__
    for key, value in config.items():
        if isinstance(value, dict):
            args[key] = Namespace(**value)
        else:
            args[key] = value

    
    # dataset = 'pmhc'
    # generative_model = 'conditional_diffusion'
    # neural_network = 'PONITA'
    # network_params = {}
    # batch_size = 64
    # lr = 1e-3
    # num_epochs = 1000

    model = Structure_Prediction_Model(
                args.dataset,
                args.generative_model,
                args.neural_network,
                args.network_params,
                args.batch_size,
                args.lr
    )

    # wandb logger
    logger = pl.loggers.WandbLogger(
        save_dir=args.logdir,
        project=args.project,
        name=args.run_name,
        entity=args.entity
    )

    # setup trainer
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        logger=logger,
        enable_progress_bar=True,
        accelerator='gpu', devices=args.gpus,
    )

    # train
    trainer.fit(model)