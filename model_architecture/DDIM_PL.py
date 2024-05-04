import pytorch_lightning as pl
import torch
import random

from collections import defaultdict
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import  DDIMScheduler
from torchvision.utils import make_grid
import torch.nn.functional as F
from model_architecture.DDPM_PL import set_timesteps_without_ratio
torch.set_float32_matmul_precision('medium')




class LightningDDIM_monai(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config=config

        self.model = DiffusionModelUNet(
            spatial_dims=config['hparams']['DiffusionModelUnet']['spatial_dims'],
            in_channels=config['hparams']['DiffusionModelUnet']['in_channels'],
            out_channels=config['hparams']['DiffusionModelUnet']['out_channels'],
            num_res_blocks=config['hparams']['DiffusionModelUnet']['num_res_blocks'],
            num_channels=config['hparams']['DiffusionModelUnet']['num_channels'],
            attention_levels=config['hparams']['DiffusionModelUnet']['attention_levels'],
            norm_num_groups=config['hparams']['DiffusionModelUnet']['norm_num_groups'],
            resblock_updown=config['hparams']['DiffusionModelUnet']['resblock_updown'],
            num_head_channels=config['hparams']['DiffusionModelUnet']['num_head_channels'],
            with_conditioning=config['hparams']['DiffusionModelUnet']['with_conditioning'],
            transformer_num_layers=config['hparams']['DiffusionModelUnet']['transformer_num_layers'],
            num_class_embeds=config['hparams']['DiffusionModelUnet']['num_class_embeds'],
            upcast_attention=config['hparams']['DiffusionModelUnet']['upcast_attention'],
            use_flash_attention=config['hparams']['DiffusionModelUnet']['use_flash_attention'],

        )
        
        
        self.scheduler = DDIMScheduler(
            num_train_timesteps=config['hparams']['DDIMScheduler']['num_train_timesteps'],
            schedule=config['hparams']['DDIMScheduler']['schedule'],
            clip_sample=config['hparams']['DDIMScheduler']['clip_sample'],
            set_alpha_to_one=config['hparams']['DDIMScheduler']['set_alpha_to_one'],
            prediction_type=config['hparams']['DDIMScheduler']['prediction_type']
        )
        self.num_inference_timesteps = config['hparams']['num_inference_timesteps']
        self.inferer = DiffusionInferer(
            scheduler=self.scheduler
        )
        self.lr = config['hparams']['learning_rate']
        self.criterion = F.mse_loss
        self.in_channels = config['hparams']['DiffusionModelUnet']['in_channels']
        self.image_h = config['hparams']['image_h']
        self.image_w = config['hparams']['image_w']
        self.outputs = defaultdict(list)
        self.batches=[]

        self.save_hyperparameters()
        

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch
        noise = torch.randn((images.shape[0], images.shape[1], images.shape[2], images.shape[3])).to(images.device)
        timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (images.shape[0],)).to(images.device)
        noise_pred = self.inferer(inputs=images, diffusion_model=self.model, noise=noise, timesteps=timesteps)

        train_loss = self.criterion(noise_pred.float(), noise.float())
        self.log("training_loss", train_loss, prog_bar=True)
        
        self.batches.append(batch)
        
        return train_loss
    

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch
        # with autocast(enabled=True):
        noise = torch.randn((images.shape[0], images.shape[1], images.shape[2], images.shape[3]), device=images.device)
        timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device).long()
        noise_pred = self.inferer(inputs=images, diffusion_model=self.model, noise=noise, timesteps=timesteps)

        val_loss = self.criterion(noise_pred.float(), noise.float())
        self.outputs[dataloader_idx].append({"val_loss": val_loss})
        return val_loss
    
    
    def on_train_epoch_end(self):
        images = self.batches[0]
        current_epoch = self.current_epoch + 1
        if current_epoch % 50 == 0:
            print(f'On training epoch:{self.current_epoch} end\n')
            int_timesteps = int(0.2 * self.inferer.scheduler.num_train_timesteps)
            timesteps = torch.tensor(int_timesteps, dtype=torch.long)

            noise = torch.randn((images.shape[0], images.shape[1], images.shape[2], images.shape[3]), device=images.device)
            noisy_image =  self.scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)

            self.scheduler.timesteps = set_timesteps_without_ratio(num_inference_steps=int_timesteps, device=images.device)
            self.scheduler.num_inference_steps = int_timesteps
            images_denoised = self.inferer.sample(input_noise=noisy_image, diffusion_model=self.model, scheduler=self.scheduler)
            
            grid_noisy = make_grid(noisy_image, nrow=4)
            grid_train_images = make_grid(images, nrow=4)
            grid_denoised_images = make_grid(images_denoised, nrow=4)
            
            
            self.logger.experiment.add_image(f"Training images", grid_train_images, current_epoch)
            self.logger.experiment.add_image(f"Noised train images", grid_noisy, current_epoch)
            self.logger.experiment.add_image(f"Denoised train images", grid_denoised_images, current_epoch)
        
        self.batches.clear()

    
    def on_validation_epoch_end(self):

        flat_outputs = []
        for lst in self.outputs.values():
            flat_outputs.extend(lst)

        avg_loss = torch.stack([x["val_loss"] for x in flat_outputs]).mean()
        current_epoch = self.current_epoch + 1
        # with autocast(enabled=True):
        if current_epoch % 50 == 0:
            print(f'On validation epoch:{self.current_epoch} end\n')
            noise = torch.randn((8, self.in_channels , self.image_h, self.image_w)).to(self.device)
            self.scheduler.set_timesteps(num_inference_steps=self.num_inference_timesteps)
            images = self.inferer.sample(input_noise=noise, diffusion_model=self.model, scheduler=self.scheduler)
            grid = make_grid(images, nrow=4)
            self.logger.experiment.add_image(f"Generated retinal image in validation epoch end", grid, current_epoch)
        self.log("validation_loss_epoch_end", avg_loss, prog_bar=True)

        self.outputs.clear()
    
   

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer