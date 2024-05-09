import os
import torch
from argparse import ArgumentParser
from utils import get_config, UK_biobank_data_module, seed_everything, load_model

from generative.networks.schedulers import DDPMScheduler,DDIMScheduler, PNDMScheduler
from generative.inferers import DiffusionInferer
from torch.cuda.amp import  autocast
from torchvision.utils import save_image, make_grid
from model_architecture import FlexibleConditionalDiffusionInferer
from generative.networks.nets import DiffusionModelUNet
torch.set_float32_matmul_precision('high')

def pipeline(config):
        #
    model = load_model(config=config)
    in_channels, image_h, image_w =  3,config['hparams']['image_h'] , config['hparams']['image_w']
    
    scheduler_DDPM = DDPMScheduler(
        num_train_timesteps=config['hparams']['num_train_timesteps']
    )
    scheduler_DDIM = DDIMScheduler(
        num_train_timesteps=config['hparams']['num_train_timesteps'],
    )
    
    inferer = FlexibleConditionalDiffusionInferer(
            scheduler=scheduler_DDIM
        )
    schedulers = [scheduler_DDIM]
    sample_images = []
    n_images = config['hparams']['number_of_samples']
    fp = config['exp']['sample_image_dir']
    os.makedirs(fp, exist_ok=True)

    for i in range(n_images):
        sample_images = []
        noise = torch.randn((1, in_channels , image_h, image_w)).to('cuda')
        noise = torch.repeat_interleave(noise,6,dim=0)
        labels = torch.arange(6)

        for j in range(len(schedulers)):
            with autocast(enabled=True):
                schedulers[j].set_timesteps(num_inference_steps=config['hparams']['num_inference_timesteps'])
                
                images = inferer.sample(input_noise=noise, diffusion_model=model, scheduler=schedulers[j], conditioning=labels)
                sample_images.append(images)
        
        sample_images = torch.concat(sample_images,dim=0)
        grid_images = make_grid(sample_images)
        save_file_path = os.path.join(fp, f"AMD_Cat_D_Gl_Mya_N_{i}.png")
        save_image(grid_images, save_file_path)
        print(f"Saved image_{i}")




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