import argparse
from argparse import Namespace
from pathlib import Path
import yaml

import os
import sys

import pytorch_lightning as pl
# from ECGDM.Experiments.Structure_prediction.lightning_module import Structure_Prediction_Model

if __name__ == "__main__":
	
    desired_directory = '/gpfs/home4/dfruhbuss/ECGDM/'
    os.chdir(desired_directory)
    sys.path.insert(0, desired_directory)
    from Experiments.Structure_prediction.lightning_module import Structure_Prediction_Model


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

    # lightning module
    model = Structure_Prediction_Model(
                args.dataset,
                args.data_dir,
                args.dataset_params,
                args.task_params,
                args.generative_model,
                args.generative_model_params,
                args.architcture,
                args.network_params,
                args.batch_size,
                args.lr,
                args.num_workers,
                args.device
    )

    # wandb logger
    logger = pl.loggers.WandbLogger(
        save_dir=args.logdir,
        project=args.project,
        name=args.run_name,
        entity=args.entity
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=Path(args.logdir, 'checkpoints'),
        filename="best-model-epoch={epoch:02d}",
        monitor="error_mol_val",
        save_top_k=1,
        save_last=True,
        mode="min",
    )

    # setup trainer
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        accelerator='gpu', devices=args.gpus,
    )

    # train
    trainer.fit(model)