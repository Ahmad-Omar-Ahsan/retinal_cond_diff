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

# def predict(filepaths, config):

#     with open(config['exp']['trial_pickle_file'], 'rb') as pickle_file:
#         trials_1000= pickle.load(pickle_file)
#     predictions = {}
#     for filepath in filepaths:
#         if filepath in trials_1000:
#             prediction_dict = trials_1000[filepath]
#             predictions[filepath] = prediction_dict['predicted_label']
#         else:
#             print(f"Warning: {filepath} not found in trials.")

#     return predictions

def load_image(file_path):
    to_tensor_transform = transforms.ToTensor()
    sample = Image.open(file_path)
    sample = sample.convert("RGB")
    sample = to_tensor_transform(sample)

    return sample

def generate_counterfactuals(config, filepaths=None):

    batch_size = config["hparams"]["batch_size"]

    with open(config['exp']['trial_pickle_file'], "rb") as pickle_file:
        trials = pickle.load(pickle_file)
    
    step_size =  config['hparams']['num_train_timesteps'] // config['hparams']['num_inference_timesteps']
    diffusion_module = Pretrained_LightningDDPM_monai.load_from_checkpoint(
         config['exp']['model_ckpt_path'], strict=False, config=config
    )
    latent_space_depth = int(config['hparams']['denoising_timestep'])
    diffusion_module.scheduler_DDIM.set_timesteps(num_inference_steps=config['hparams']['num_inference_timesteps'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = config['hparams']['num_classes']
    diffusion_module.scheduler_DDIM.clip_sample = False
    diffusion_module.to(device=device)
    diffusion_module.scheduler_DDIM.to(device=device)

    selected_trials = {k: v for k, v in trials.items() if (filepaths is None or k in filepaths)}
    filepaths_list = list(selected_trials.keys())
   
    for start_idx in range(0, len(filepaths_list), batch_size):
        batch_paths = filepaths_list[start_idx:start_idx + batch_size]
        batch_labels, batch_preds, batch_images = [], [], []

        for filepath in batch_paths:
            values = selected_trials[filepath]
            label = torch.tensor(np.array(values["test_label"]), device=device)
            pred = torch.tensor(np.array(values["predicted_label"]), device=device)
            image = load_image(filepath).to(device)

            batch_labels.append(label)
            batch_preds.append(pred)
            batch_images.append(image)

        batch_labels = torch.stack(batch_labels)      
        batch_preds = torch.stack(batch_preds)       
        batch_images = torch.stack(batch_images)  

        B, C, H, W = batch_images.shape

        start = time.time()
        with torch.no_grad():
            current_img = batch_images.unsqueeze(1).repeat(1, num_classes, 1, 1, 1)  
            current_img = current_img.view(B * num_classes, C, H, W).to(device)
        
            cond_labels = torch.arange(num_classes).to(device).repeat(B)             
            pred_repeated = batch_preds.unsqueeze(1).repeat(1, num_classes).view(-1) 
            
            for t in range(0, latent_space_depth + step_size, step_size):
                timesteps = torch.tensor([t]).to(device).repeat(B * num_classes)
                model_output = diffusion_module.model(
                    current_img, timesteps=torch.tensor([t]).to(device).repeat(B*num_classes), class_labels=pred_repeated
                )
                current_img, _ = diffusion_module.scheduler_DDIM.reversed_step(
                    model_output, t, current_img
                )
           
            

            for t in np.arange(config['hparams']['denoising_timestep'], -step_size, -step_size):
                timesteps = torch.tensor([t]).to(device).repeat(num_classes * B)
                model_output = diffusion_module.model(
                    current_img, timesteps=timesteps, class_labels=cond_labels
                )
                current_img, _ = diffusion_module.scheduler_DDIM.step(
                    model_output, t, current_img
                )
            current_img_multiple = current_img.view(B, num_classes, C, H, W)
            for idx, filepath in enumerate(batch_paths):
                filename = os.path.splitext(os.path.basename(filepath))[0]
                image_path = os.path.join(
                    config['exp']['counterfactual_dir'],
                    f"A_{batch_labels[idx].cpu().numpy()}_P_{batch_preds[idx].cpu().numpy()}_{filename}.png"
                )

                if os.path.exists(image_path):
                    print(f"Path exists: {image_path}")
                    continue

                concat_images = torch.cat([batch_images[idx:idx+1], current_img_multiple[idx]], dim=0)
                grid = make_grid(concat_images, nrow=num_classes + 1)
                save_image(grid, fp=image_path)
                print(f"Saved: {image_path}")

            end = time.time()
            print(f"Elapsed time: {end-start}s.")


            


        
        




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
