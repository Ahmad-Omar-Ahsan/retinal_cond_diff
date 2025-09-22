import os
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
import torch
from utils import get_config, UK_biobank_data_module, seed_everything, FakeData_lightning, Retinal_Cond_Lightning_Split, Pickle_Lightning
from model_architecture import LightningDDPM_monai,  Pretrained_LightningDDPM_monai,Conditional_DDIM_monai,MLP_classifier
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
import copy
import numpy as np
import time
from torchvision import transforms
from PIL import Image 
import pickle

def predict(filepath, config):
    with open(config['exp']['trial_pickle_file'], 'rb') as pickle_file:
        trials_1000= pickle.load(pickle_file)
    prediction_dict = trials_1000[filepath]
    prediction = prediction_dict['predicted_label']
    return prediction

def load_image(file_path):
    to_tensor_transform = transforms.ToTensor()
    sample = Image.open(file_path)
    sample = sample.convert("RGB")
    sample = to_tensor_transform(sample)

    return sample

def generate_counterfactuals(config):

    with open(config['exp']['trial_pickle_file'], "rb") as pickle_file:
        trials = pickle.load(pickle_file)
    
    step_size =  config['hparams']['num_train_timesteps'] // config['hparams']['num_inference_timesteps']
    diffusion_module = Pretrained_LightningDDPM_monai.load_from_checkpoint(
         config['exp']['model_ckpt_path'], strict=False, config=config
    )
    latent_space_depth = int(config['hparams']['denoising_timestep'])
    diffusion_module.scheduler_DDIM.set_timesteps(num_inference_steps=config['hparams']['num_inference_timesteps'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    count = 0
    num_classes = config['hparams']['num_classes']
    diffusion_module.scheduler_DDIM.clip_sample = False
    diffusion_module.scheduler_DDIM.clip_sample_min = 0.0
    diffusion_module.scheduler_DDIM.clip_sample_min = 1.0
   
    for key, values in trials.items():
        filepath = key
        label = values["test_label"]
        pred = values["predicted_label"]
        with torch.no_grad():

            start = time.time()
            label = torch.tensor(np.array(label), device=device)

            prediction = torch.tensor(np.array(pred), device=device)
            
            filename = filepath.split('/')[-1].replace(".png", "")
            image_path = os.path.join(config['exp']['counterfactual_dir'], f"A_{label.cpu().numpy()}_P_{prediction.cpu().numpy()}_reconstructed_{filename}_{count}.png")
            

            
            print(f"Label:{label.cpu().numpy()}, Prediction: {prediction.cpu().numpy()}")
            if os.path.exists(image_path):
                print(f"Path exists: {image_path}")
                continue
            image = load_image(file_path=filepath)
            image = image.to(device)
            diffusion_module.model = diffusion_module.model.to(device)
            current_img = image.unsqueeze(0).repeat(num_classes, 1, 1, 1).to(device)
            pred = prediction.unsqueeze(0).repeat(num_classes)

            for i in range(0, latent_space_depth + step_size, step_size):
                t = i
                model_output = diffusion_module.model(
                    current_img, timesteps=torch.tensor([t]).to(device), class_labels=pred
                )
                current_img, _ = diffusion_module.scheduler_DDIM.reversed_step(
                    model_output, t, current_img
                )
            current_img_multiple = current_img
            conditions = torch.arange(num_classes).to(device)

            for t in np.arange(config['hparams']['denoising_timestep'], -step_size, -step_size):
                timesteps = torch.tensor([t]).to(device).repeat(num_classes)
                model_output = diffusion_module.model(
                    current_img_multiple, timesteps=timesteps, class_labels=conditions
                )
                current_img_multiple, _ = diffusion_module.scheduler_DDIM.step(
                    model_output, t, current_img_multiple
                )

            concat_images = torch.cat([image.unsqueeze(0), current_img_multiple], dim=0)  # Adjusted for batch
            grid = make_grid(concat_images)
            save_image(grid, fp=image_path)
            end = time.time()
            print(f"Elapsed time: {end-start}s.")
            print(f"Saved: {image_path}")


            


        
        




def main(args):
    config = get_config(args.conf)
    seed_everything(config["hparams"]["seed"])
    os.makedirs(config['exp']['counterfactual_dir'],exist_ok=True)
    generate_counterfactuals(config)



if __name__=="__main__":
    parser = ArgumentParser("Driver code.")
    parser.add_argument(
        "--conf", type=str, required=True, help="Path to config.yaml file."
    )
    args = parser.parse_args()

    main(args)
