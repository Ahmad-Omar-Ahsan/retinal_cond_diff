import os
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

from utils import get_config, UK_biobank_data_module, seed_everything, FakeData_lightning, Retinal_Cond_Lightning, load_model
from model_architecture import LightningDDPM_monai, LightningDDIM_monai, LightningDDPMDDIM_monai, Pretrained_LightningDDPM_monai,Conditional_DDIM_monai
from generative.networks.nets import DiffusionModelUNet


def pipeline(config):
    logger = TensorBoardLogger(config['exp']['logdir'], name=config['exp']["exp_name"])
    
    dm = Retinal_Cond_Lightning(
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
        # profiler='pytorch',
        accumulate_grad_batches=8
    )
    if config['exp']['training_type'] == 'scratch':
        DDIM_lightning = Conditional_DDIM_monai(config=config)
        trainer.fit(model=DDIM_lightning, datamodule=dm)
    elif config['exp']['training_type'] == 'pretrained':
        # checkpoint = torch.load(config['exp']['model_ckpt_path'])
        # unet_weights = {k:v for k,v in checkpoint['state_dict'].items() if k.startswith('model')}
        # unet_weights = {k.replace('model.',''):v for k,v in unet_weights.items()}
        # Pretrained_DDPM_lightning = Pretrained_LightningDDPM_monai(config=config, unet_weights=unet_weights)
        Pretrained_DDPM_lightning = Pretrained_LightningDDPM_monai.load_from_checkpoint(config['exp']['model_ckpt_path'], strict=False, config=config)
        trainer.fit(model=Pretrained_DDPM_lightning, datamodule=dm)
    elif config['exp']['training_type'] == 'test':
        Pretrained_DDPM_lightning = Pretrained_LightningDDPM_monai(config=config)
        trainer.predict(model=Pretrained_DDPM_lightning, datamodule=dm)
    # model = load_model(config=config)
    # DDPM_lightning = Pretrained_LightningDDPM_monai(config=config,model=model)
    
    

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



