import os
import pytorch_lightning as pl

from lightning.pytorch.loggers import TensorBoardLogger
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

from utils import get_config, UK_biobank_data_module, seed_everything, FakeData_lightning
from model_architecture import LightningDDPM_monai, LightningDDIM_monai, LightningDDPMDDIM_monai



def pipeline(config):
    logger = TensorBoardLogger(config['exp']['logdir'], name=config['exp']["exp_name"])
    
    dm = UK_biobank_data_module(
        config=config
    )
    # dm = FakeData_lightning(config=config)
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir'], config['hparams']['scheduler_type']),
                                              monitor='validation_loss_epoch_end',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              )
    trainer = pl.Trainer(
        logger=logger,
        log_every_n_steps=config['exp']['log_every_n_steps'],
        devices=config['exp']['device'],
        min_epochs = config['hparams']['min_epochs'],
        max_epochs = config['hparams']['max_epochs'],
        num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
        accelerator=config['exp']['accelerator'],
        callbacks=[checkpoint_callback],
        precision='16-mixed',
        profiler='pytorch',
        accumulate_grad_batches=8
    )
    if config['hparams']['scheduler_type'] == 'DDPM':
        DDPM_lightning = LightningDDPM_monai(config=config)
        trainer.fit(model=DDPM_lightning, datamodule=dm)
    elif config['hparams']['scheduler_type'] == 'DDIM':
        DDIM_lightning = LightningDDIM_monai(config=config)
        trainer.fit(model=DDIM_lightning, datamodule=dm)
    else:
        DDPM_DDIM_lightning = LightningDDPMDDIM_monai(config=config)
        trainer.fit(model=DDPM_DDIM_lightning, datamodule=dm)

def main(args):
    config = get_config(args.conf)
    seed_everything(config["hparams"]["seed"])
    pipeline(config)

if __name__=="__main__":
    parser = ArgumentParser("Driver code.")
    parser.add_argument(
        "--conf", type=str, required=True, help="Path to config.yaml file."
    )
    args = parser.parse_args()

    main(args)



