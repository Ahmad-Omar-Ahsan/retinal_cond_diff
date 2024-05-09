import torch
import numpy as np
from torch import nn
import random
import os
from generative.networks.nets import DiffusionModelUNet
from model_architecture import LightningDDPM_monai, LightningDDPMDDIM_monai, Pretrained_LightningDDPM_monai

def seed_everything(seed: str) -> None:
    """Set manual seed.
    Args:
        seed (int): Supplied seed.
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Set seed {seed}")


def count_params(model: nn.Module) -> int:
    """Counts number of parameters in a model.
    Args:
        model (torch.nn.Module): Model instance for which number of params is to be counted.
    Returns:
        int: Parameter count.
    """

    return sum(map(lambda p: p.data.numel(), model.parameters()))


def load_model_ckpt(model_ckpt_path: str, config: dict) -> nn.Module:
   
    
    if config['hparams']['pretrained_scheduler_type'] == 'DDPM':
        l_module = LightningDDPM_monai.load_from_checkpoint(model_ckpt_path)
    elif config['hparams']['pretrained_scheduler_type'] =='DDPM_DDIM':
        l_module = LightningDDPMDDIM_monai.load_from_checkpoint(model_ckpt_path)
    elif config['hparams']['pretrained_scheduler_type'] == 'conditional_DDPM':
        checkpoint = torch.load(model_ckpt_path)
        unet_weights = {k:v for k,v in checkpoint['state_dict'].items() if k.startswith('model')}
        unet_weights = {k.replace('model.',''):v for k,v in unet_weights.items()}
        l_module = LightningDDPMDDIM_monai.load_from_checkpoint("/home/ahmad/ahmad_experiments/retinal_cond_diff/epoch=161-step=82134.ckpt")
        pl_module = Pretrained_LightningDDPM_monai(config=config, model=l_module.model)
        pl_module.model.load_state_dict(unet_weights)
        return pl_module.model
        # l_module.model.load_state_dict(unet_weights)
    
    return l_module.model


def load_model(config: dict) -> nn.Module:
    """ Load model with hparams

    Args:
        config (dict): Config file containing hyperparameters

    Returns:
        nn.Module: Model
    """
    if 'DiffusionModelUnet' in config['hparams']:
        model = DiffusionModelUNet(
            spatial_dims=config['hparams']['DiffusionModelUnet']['spatial_dims'],
            in_channels=config['hparams']['DiffusionModelUnet']['in_channels'],
            out_channels=config['hparams']['DiffusionModelUnet']['out_channels'],
            num_channels=config['hparams']['DiffusionModelUnet']['num_channels'],
            attention_levels=config['hparams']['DiffusionModelUnet']['attention_levels'],
            num_res_blocks=config['hparams']['DiffusionModelUnet']['num_res_blocks'],
            num_head_channels=config['hparams']['DiffusionModelUnet']['num_head_channels'],
            num_class_embeds=config['hparams']['DiffusionModelUnet']['num_class_embeds']
        )
    model = load_model_ckpt(model_ckpt_path=config['exp']['model_ckpt_path'],config=config)
    return model