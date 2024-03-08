import argparse
from argparse import Namespace
import yaml

import pytorch_lightning as pl
from ECGDM.Experiments.Structure_prediction.lightning_module import Structure_Prediction_Model

if __name__ == "__main__":

    # read in config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.save_load(f)

    args_dict = args.__dict__
    for key, value in config.items():
        if isinstance(value, dict):
            args[key] = Namespace(**value)
        else:
            args[key] = value

    # lightning module
    model = Structure_Prediction_Model(
                args.dataset,
                args.data_dir,
                args.dataset_params,
                args.generative_model,
                args.generative_model_params,
                args.neural_network,
                args.network_params,
                args.batch_size,
                args.lr,
                args.device
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