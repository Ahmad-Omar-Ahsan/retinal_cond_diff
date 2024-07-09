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


def predict(config, test_image, diffusion_module, device):
    classes = torch.arange(6).to(device)
    class_errors = []
    # test_image = torch.unsqueeze(test_image, dim=0).to(device)
    test_image = torch.repeat_interleave(test_image, 6
                                            ,dim=0)
    error = [0,0,0,0,0,0]
    

    for r in range(config['hparams']['runs']):
        timesteps = torch.randint(0, diffusion_module.scheduler.num_train_timesteps,(1,),device=device).long()
        timesteps=torch.repeat_interleave(timesteps,diffusion_module.batch_size,dim=0)

        noise = torch.randn((1, diffusion_module.in_channels , diffusion_module.image_h, diffusion_module.image_w)).to(device)
        noise = torch.repeat_interleave(noise,6,dim=0)
        
        for c in range(0, 6, 6):
            conditions = classes[c: c+6]
            output = diffusion_module.inferer(inputs=test_image, diffusion_model=diffusion_module.model, noise=noise, timesteps=timesteps, conditioning=conditions)
            value = diffusion_module.criterion(noise, output,reduction='none').mean(dim=(1,2,3)).view(-1).to(device)
            error[c: 6] = value.cpu().numpy()
        
        class_errors.append(copy.deepcopy(error))

    
    np_class_errors = np.array(class_errors)
    mean_error_classes = np.mean(np_class_errors, axis=0)
    min_error_index = np.argmin(mean_error_classes, axis=0)
    return min_error_index


def generate_counterfactuals(config):
    dm = Retinal_Cond_Lightning_Split(
            config=config
    )
    dm.setup('test')
    step_size =  config['hparams']['num_train_timesteps'] // config['hparams']['num_inference_timesteps']
    diffusion_module = Pretrained_LightningDDPM_monai.load_from_checkpoint(config['exp']['model_ckpt_path'], strict=False, config=config)
    latent_space_depth = int(config['hparams']['denoising_timestep'])
    progress_bar = tqdm(range(0, latent_space_depth+step_size, step_size))
    diffusion_module.scheduler_DDIM.set_timesteps(num_inference_steps=config['hparams']['num_inference_timesteps'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    count = 0
    
    
    with torch.no_grad():
        for image, label in dm.test_dataloader():
            label = label.to(device)
            image = image.to(device)
            current_img = image.to(device)
            pred = predict(config, image, diffusion_module, device)
            print(f"Label:{label.item()}, Prediction: {pred}")
            diffusion_module.model = diffusion_module.model.to(device)
            diffusion_module.scheduler_DDIM.clip_sample = False
            pred = torch.as_tensor([pred]).to(device)
            for i in progress_bar:
                t = i
                
                model_output = diffusion_module.model(current_img, timesteps=torch.Tensor((t,)).to(device), class_labels=pred).to(device)
                current_img, _ = diffusion_module.scheduler_DDIM.reversed_step(model_output, t, current_img)

            latent_img = current_img
            current_img_multiple = torch.repeat_interleave(current_img, diffusion_module.num_classes
                                                    ,dim=0)

            conditions = torch.arange(6).to(device)
            diffusion_module.scheduler_DDIM.clip_sample = False

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
            concat_images = torch.concat((image, latent_img, current_img_multiple),dim=0)
            grid = make_grid(concat_images)
            image_path = os.path.join(config['exp']['counterfactual_dir'],f"Label_{label.item()}_pred_{pred.item()}_count_{count}.png")
            save_image(grid, fp=image_path)
            count += 1

            


        
        




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
