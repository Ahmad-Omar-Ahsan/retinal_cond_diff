import os
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
import torch
from utils import get_config, UK_biobank_data_module, seed_everything, FakeData_lightning, Retinal_Cond_Lightning_Split, load_model, Pickle_Lightning
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
    
    
    with torch.no_grad():
        for image, label in dm.test_dataloader():
            start = time.time()
            label = label.to(device)
            image = image.to(device)
            current_img = image.to(device)
            file_path = file_paths[count][0]
            count += 1
            pred = predict(file_path, config)
            pred = torch.as_tensor([pred]).to(device)
            image_path = os.path.join(config['exp']['counterfactual_dir'], f"A_{label.item()}_P_{pred.item()}_reconstructed_{count}.png")
            print(f"Label:{label.item()}, Prediction: {pred.item()}")
            if os.path.exists(image_path):
                print(f"Path exists: {image_path}")
                continue
            diffusion_module.model = diffusion_module.model.to(device)
            diffusion_module.scheduler_DDIM.clip_sample = False
            for i in progress_bar:
                t = i
                
                model_output = diffusion_module.model(current_img, timesteps=torch.Tensor((t,)).to(device), class_labels=pred).to(device)
                current_img, _ = diffusion_module.scheduler_DDIM.reversed_step(model_output, t, current_img)

            latent_img = current_img
            current_img_multiple = torch.repeat_interleave(current_img, diffusion_module.num_classes
                                                    ,dim=0)

            conditions = torch.arange(6).to(device)
            diffusion_module.scheduler_DDIM.clip_sample = False
            # label_dir = os.path.join(config['exp']['counterfactual_dir'], str(label.item()))
            # os.makedirs(label_dir, exist_ok=True)

            for t in np.arange(config['hparams']['denoising_timestep'], -step_size, -step_size):
                timesteps = torch.Tensor((t,)).to(device)
                timesteps=torch.repeat_interleave(timesteps,6,dim=0)
                model_output = diffusion_module.model(current_img_multiple, timesteps=timesteps, class_labels=conditions).to(device)
                amd, _ = diffusion_module.scheduler_DDIM.step(model_output[0].unsqueeze(dim=0), t, current_img_multiple[0].unsqueeze(dim=0))
                cataract, _ = diffusion_module.scheduler_DDIM.step(model_output[1].unsqueeze(dim=0), t, current_img_multiple[1].unsqueeze(dim=0))
                dr, _ = diffusion_module.scheduler_DDIM.step(model_output[2].unsqueeze(dim=0), t, current_img_multiple[2].unsqueeze(dim=0))
                glaucoma, _ = diffusion_module.scheduler_DDIM.step(model_output[3].unsqueeze(dim=0), t, current_img_multiple[3].unsqueeze(dim=0))
                myopia, _ = diffusion_module.scheduler_DDIM.step(model_output[4].unsqueeze(dim=0), t, current_img_multiple[4].unsqueeze(dim=0))
                normal, _ = diffusion_module.scheduler_DDIM.step(model_output[5].unsqueeze(dim=0), t, current_img_multiple[5].unsqueeze(dim=0))
                current_img_multiple = torch.concat((amd,cataract,dr,glaucoma,myopia,normal))
            # images = diffusion_module.inferer.sample(input_noise=current_img_multiple, diffusion_model=diffusion_module.model, scheduler=diffusion_module.scheduler_DDIM, conditioning=conditions)
            # amd_image_path = os.path.join(config['exp']['counterfactual_dir'],f"Label_{label.item()}_amd_count_{count}.png")
            # cataract_image_path = os.path.join(config['exp']['counterfactual_dir'],f"Label_{label.item()}_cataract_count_{count}.png")
            # dr_image_path = os.path.join(config['exp']['counterfactual_dir'],f"Label_{label.item()}_dr_count_{count}.png")
            # glaucoma_image_path = os.path.join(config['exp']['counterfactual_dir'],f"Label_{label.item()}_glaucoma_count_{count}.png")
            # myopia_image_path = os.path.join(config['exp']['counterfactual_dir'],f"Label_{label.item()}_myopia_count_{count}.png")
            # normal_image_path = os.path.join(label_dir,f"Label_{label.item()}_normal_count_{count}.png")
            # original_image_path = os.path.join(label_dir,f"Label_{label.item()}_original_image_count_{count}.png")
            # # save_image(amd,fp=amd_image_path)
            # # save_image(cataract,fp=cataract_image_path)
            # # save_image(dr,fp=dr_image_path)
            # # save_image(glaucoma,fp=glaucoma_image_path)
            # # save_image(myopia, fp=myopia_image_path)
            # save_image(normal,fp=normal_image_path)
            # save_image(image, fp=original_image_path)

            concat_images = torch.concat((image, latent_img, current_img_multiple),dim=0)
            grid = make_grid(concat_images)
            
            
            save_image(grid, fp=image_path)
            end = time.time()
            print(f"Elasped time: {end-start}s.")
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
