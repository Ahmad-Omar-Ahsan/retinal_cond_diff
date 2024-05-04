import os
import torch
from argparse import ArgumentParser
from utils import get_config, UK_biobank_data_module, seed_everything, load_model
import glob
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from generative.inferers import DiffusionInferer
from torch.cuda.amp import  autocast
from torchvision.utils import save_image,make_grid
torch.set_float32_matmul_precision('high')
from model_architecture import set_timesteps_without_ratio
from utils import UK_biobank_retinal
from torch.utils.data import DataLoader
from tqdm import tqdm

def pipeline(config):
    model = load_model(config=config)
    model = model.to(device='cuda')
    # in_channels, image_h, image_w =  3,config['hparams']['image_h'] , config['hparams']['image_w']
    scheduler_DDIM = DDIMScheduler(
        num_train_timesteps=config['hparams']['num_train_timesteps']
        )
    scheduler_DDPM = DDPMScheduler(
        num_train_timesteps=config['hparams']['num_train_timesteps']
    )
    inferer = DiffusionInferer(
            scheduler=scheduler_DDIM
        )
    
    data_dir = config['exp']['data_dir']
    file_pattern = f"*.{config['exp']['image_extension']}"
    files_path = os.path.join(data_dir, file_pattern)
    sample_list = glob.glob(files_path)
    ds = UK_biobank_retinal(sample_list=sample_list)
    fp = config['exp']['denoised_img_dir']
    count = config['hparams']['number_of_samples']
    data_loader = DataLoader(ds, batch_size=1)
    os.makedirs(fp, exist_ok=True)

    
    for i, image in enumerate(data_loader):
        image = image.to(device='cuda')
        with autocast(enabled=True):
            
            int_timesteps = config['hparams']['denoising_timestep']
            timesteps = torch.tensor(int_timesteps, dtype=torch.long, device=image.device)
            noise = torch.randn((image.shape[0], image.shape[1], image.shape[2], image.shape[3]), device=image.device)

            # noisy_image =  scheduler_DDIM.add_noise(original_samples=image, noise=noise, timesteps=timesteps)
            noisy_image = image
            progress_bar = tqdm(range(int_timesteps))
            for t in progress_bar:
                with torch.no_grad():
                    model_output = model(
                        noisy_image, timesteps=torch.Tensor((t,)).to(noise.device), context=None
                    )
                noisy_image, _ = scheduler_DDIM.reversed_step(model_output=model_output,timestep=i, sample=model_output)
            scheduler_DDIM.timesteps = set_timesteps_without_ratio(num_inference_steps=int_timesteps, device=image.device)
            _, intermediates = inferer.sample(input_noise=noisy_image, diffusion_model=model, scheduler=scheduler_DDIM, save_intermediates=True)

        intermediates_tensor = torch.concat(intermediates, dim=0)
        grids = torch.concat((image, noisy_image,intermediates_tensor),dim=0)
        grid = make_grid(grids)
        if i == count:
            break
        save_file_path = os.path.join(fp, f"_{i}.png")
        save_image(grid, save_file_path)
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