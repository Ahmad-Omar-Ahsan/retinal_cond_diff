
import torch
import os
from utils import get_config, UK_biobank_data_module, seed_everything

from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from generative.inferers import DiffusionInferer
from torch.cuda.amp import  autocast
from torchvision.utils import save_image,make_grid
from torchvision import transforms
torch.set_float32_matmul_precision('medium')
from model_architecture import set_timesteps_without_ratio
from utils import UK_biobank_retinal
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image


def main():
    config = get_config("/home/ahmad/ahmad_experiments/retinal_cond_diff/conf.yaml")
    seed_everything(config["hparams"]["seed"])
    img_path = "/home/ahmad/ahmad_experiments/retinal_data/noising_sample/148.png"
    sample = Image.open(img_path)
    sample = sample.convert("RGB")
    transform =transforms.Compose([transforms.ToTensor()])
    image = transform(sample)
    filename = (img_path.split('/')[-1]).replace(".png","")
    timesteps =[100,200,300,400, 500, 600,700, 800,900, 1000]

    scheduler_DDPM = DDPMScheduler(
        num_train_timesteps=config['hparams']['num_train_timesteps']
    )
    noisy_samples = [image]
    noise = torch.randn((image.shape[0], image.shape[1], image.shape[2]), device=image.device)
    for i in timesteps:
        t = torch.tensor(i-1, dtype=torch.long, device=image.device)
        noised_image = scheduler_DDPM.add_noise(original_samples=image, noise=noise, timesteps=t)
        noisy_samples.append(noised_image)
    
    concatenated_image = torch.stack(noisy_samples, dim=0)
    image = torch.unsqueeze(image, dim=0)
    grids = torch.concat((image, concatenated_image),dim=0)
    print(grids.shape)
    grid = make_grid(grids)
    os.makedirs(config['exp']['noised_img_dir'],exist_ok=True)
    fp = config['exp']['noised_img_dir']
    save_file_path = os.path.join(fp, f"{filename}_noised_image.png")
    save_image(grid, save_file_path)
    print(f"Saved image_{i}")


    for i, noisy_sample in enumerate(noisy_samples):
        # Create a grid with the original image and the current noisy sample
        grid =  torch.unsqueeze(noisy_sample, dim=0)

        # Ensure the directory exists
        # os.makedirs(config['exp']['noised_img_dir'], exist_ok=True)

        # Create the file path for the current image
        save_file_path = os.path.join(config['exp']['noised_img_dir'], f"{filename}_noised_image_{i}.png")

        # Save the image
        save_image(grid, save_file_path)

        print(f"Saved image_{i}")

    print("All images saved successfully!")


if __name__ == "__main__":
    main()