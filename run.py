import os
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
import torch
from utils import get_config, UK_biobank_data_module, seed_everything, FakeData_lightning, Retinal_Cond_Lightning_Split, load_model, Pickle_Lightning
from model_architecture import LightningDDPM_monai,  Pretrained_LightningDDPM_monai,Conditional_DDIM_monai,MLP_classifier
from generative.networks.nets import DiffusionModelUNet


def pipeline(config):
    logger = TensorBoardLogger(config['exp']['logdir'], name=config['exp']["exp_name"])
    
    
    # dm = FakeData_lightning(config=config)
    if config['exp']['training_type'] == 'scratch_UKBB':
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="diffusion-{epoch:02d}-{step}-{val_loss:.2f}"
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
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        dm = UK_biobank_data_module(
            config=config
        )
        DDPM_lightning = LightningDDPM_monai(config=config)
        trainer.fit(model=DDPM_lightning, datamodule=dm)
    elif config['exp']['training_type'] == 'scratch_conditional':
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="diffusion-{epoch:02d}-{step}-{val_loss:.2f}"
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
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        dm = Retinal_Cond_Lightning_Split(
            config=config
        )
        DDIM_lightning = Conditional_DDIM_monai(config=config)
        trainer.fit(model=DDIM_lightning, datamodule=dm)
    elif config['exp']['training_type'] == 'pretrained':
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="diffusion-{epoch:02d}-{step}-{val_loss:.2f}"
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
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        dm = Retinal_Cond_Lightning_Split(
            config=config
        )
        Pretrained_DDPM_lightning = Pretrained_LightningDDPM_monai.load_from_checkpoint(config['exp']['model_ckpt_path'], strict=False, config=config)
        trainer.fit(model=Pretrained_DDPM_lightning, datamodule=dm)
    elif config['exp']['training_type'] == 'test':
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="diffusion-{epoch:02d}-{step}-{val_loss:.2f}"
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
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        os.makedirs(config['exp']['csv_dir'],exist_ok=True)
        dm = Retinal_Cond_Lightning_Split(
            config=config
        )
        Pretrained_DDPM_lightning = Pretrained_LightningDDPM_monai.load_from_checkpoint(config['exp']['model_ckpt_path'], strict=False, config=config)
        trainer.test(model=Pretrained_DDPM_lightning, datamodule=dm)
    elif config['exp']['training_type'] == "mlp":
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="mlp-{epoch:02d}-{step}-{val_loss:.2f}"
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
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        simple_mlp = MLP_classifier(config=config)
        dm = Pickle_Lightning(config=config)
        trainer.fit(model=simple_mlp, datamodule=dm)
        trainer.test(model=simple_mlp, datamodule=dm)
    elif config['exp']['training_type'] == "cond_diff_continue":
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="diffusion-{epoch:02d}-{step}-{val_loss:.2f}"
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
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        os.makedirs(config['exp']['csv_dir'],exist_ok=True)
        dm = Retinal_Cond_Lightning_Split(
            config=config
        )
        Pretrained_DDPM_lightning = Pretrained_LightningDDPM_monai(config=config)
        trainer.fit(model=Pretrained_DDPM_lightning, datamodule=dm, ckpt_path=config['exp']['model_ckpt_path'])
    
    

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



