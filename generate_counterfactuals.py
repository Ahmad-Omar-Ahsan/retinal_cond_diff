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
import pickle

def predict(filepath, config):
    with open(config['exp']['trial_pickle_file'], 'rb') as pickle_file:
        trials_1000= pickle.load(pickle_file)
    prediction_dict = trials_1000[filepath]
    prediction = prediction_dict['predicted_label']
    return prediction

def generate_counterfactuals(config):
    dm = Retinal_Cond_Lightning_Split(
            config=config
    )
    dm.setup('test')
    file_paths = dm.test.imgs
    step_size =  config['hparams']['num_train_timesteps'] // config['hparams']['num_inference_timesteps']
    diffusion_module = Pretrained_LightningDDPM_monai.load_from_checkpoint(config['exp']['model_ckpt_path'], strict=False, config=config)
    latent_space_depth = int(config['hparams']['denoising_timestep'])
    progress_bar = tqdm(range(0, latent_space_depth+step_size, step_size))
    diffusion_module.scheduler_DDIM.set_timesteps(num_inference_steps=config['hparams']['num_inference_timesteps'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    count = 0
    num_classes = config['hparams']["num_classes"]
    
    
    with torch.no_grad():
        for images, labels in dm.test_dataloader():
            start = time.time()
            labels = labels.to(device)
            images = images.to(device)
            current_img = images.to(device)
            file_paths_batch = [file_path[0] for file_path in file_paths[count:count + len(images)]]
            count += len(images)

             # Batch prediction
            predictions = [predict(file_path, config) for file_path in file_paths_batch]
            predictions = torch.tensor(predictions, device=device)

            filenames = [
                (file_path[0].split('/')[-1]).replace(".png", "")
                for file_path in file_paths_batch
            ]
            image_paths = [
                os.path.join(config['exp']['counterfactual_dir'], f"A_{label.item()}_P_{pred.item()}_reconstructed_{filename}_{count}.png")
                for label, pred, filename in zip(labels, predictions, filenames)
            ]
            for image_path, label, pred, image in zip(image_paths, labels, predictions, images):
                print(f"Label:{label.item()}, Prediction: {pred.item()}")
                if os.path.exists(image_path):
                    print(f"Path exists: {image_path}")
                    continue

                diffusion_module.model = diffusion_module.model.to(device)
                diffusion_module.scheduler_DDIM.clip_sample = False

                current_img = image.unsqueeze(0).repeat(num_classes, 1, 1, 1)  # Batch-aware repeat
                pred = pred.unsqueeze(0).repeat(num_classes)

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
