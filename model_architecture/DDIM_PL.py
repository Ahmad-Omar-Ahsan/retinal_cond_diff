import lightning.pytorch as pl
import torch
import random

from collections import defaultdict
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import  DDIMScheduler,DDPMScheduler
from torchvision.utils import make_grid
import torch.nn.functional as F
from model_architecture.DDPM_PL import set_timesteps_without_ratio
from .Custom_Inferer import FlexibleConditionalDiffusionInferer
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle




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
    

class UnNormalize(object):
    def __init__(self, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
        """
        Args:
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.
        """
        self.mean = torch.tensor(mean).view(1, 1, -1, 1, 1)  # shape (1, C, 1, 1)
        self.std = torch.tensor(std).view(1, 1, -1, 1, 1)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) or (B, C, H, W) to be unnormalized.
        Returns:
            Tensor: Unnormalized image(s).
        """

        self.mean = self.mean.to(device=tensor.device, dtype=tensor.dtype)
        self.std = self.std.to(device=tensor.device, dtype=tensor.dtype)

        if not torch.is_tensor(tensor):
            raise TypeError(f"Input tensor expected, got {type(tensor)}")
        
        if tensor.ndim == 3:
            # Single image (C, H, W)
            return tensor * self.std.view(-1, 1, 1) + self.mean.view(-1, 1, 1)
        elif tensor.ndim == 5:
            # Batch with multiple classes (B, N_classes, C, H, W)
            return tensor * self.std + self.mean
        elif tensor.ndim == 4:
            # Regular batch (B, C, H, W)
            mean = self.mean.view(1, -1, 1, 1)
            std = self.std.view(1, -1, 1, 1)
            return tensor * std + mean
        else:
            raise ValueError(f"Expected tensor with 3, 4, or 5 dims, got {tensor.ndim}")



class Conditional_DDIM_monai(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.seed = config['hparams']['seed']

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
        self.num_classes = config['hparams']['num_classes']

        self.scheduler_DDPM = DDPMScheduler(
            num_train_timesteps=config['hparams']['DDPMScheduler']['num_train_timesteps'],
            schedule=config['hparams']['DDPMScheduler']['schedule'],
            variance_type=config['hparams']['DDPMScheduler']['variance_type'],
            clip_sample=config['hparams']['DDPMScheduler']['clip_sample'],
            prediction_type=config['hparams']['DDPMScheduler']['prediction_type']
        )
        self.scheduler_DDIM = DDIMScheduler(
            num_train_timesteps=config['hparams']['DDIMScheduler']['num_train_timesteps'],
            schedule=config['hparams']['DDIMScheduler']['schedule'],
            clip_sample=config['hparams']['DDIMScheduler']['clip_sample'],
            set_alpha_to_one=config['hparams']['DDIMScheduler']['set_alpha_to_one'],
            prediction_type=config['hparams']['DDIMScheduler']['prediction_type']
        )
        self.num_inference_timesteps = config['hparams']['num_inference_timesteps']
        self.inferer = FlexibleConditionalDiffusionInferer(
            scheduler=self.scheduler
        )
        self.lr = config['hparams']['learning_rate']
        self.criterion = F.mse_loss
        self.in_channels = config['hparams']['DiffusionModelUnet']['in_channels']
        self.image_h = config['hparams']['image_h']
        self.image_w = config['hparams']['image_w']
        
        self.batches = []  
        self.classes = torch.arange(self.num_classes)

        noise = torch.randn((1, self.in_channels , self.image_h, self.image_w))
        noise = torch.repeat_interleave(noise,self.num_classes,dim=0)
        self.noise = noise
        self.runs = config['hparams']['runs']
        # self.errors_index = 
        self.test_outputs = []
        self.class_acc = []
        self.latent_space_depth = int(config['hparams']['denoising_timestep'])
        self.step_size =  config['hparams']['num_train_timesteps'] // config['hparams']['num_inference_timesteps']
        self.save_hyperparameters(ignore=["unet_weights",'file_path_labels'])

        self.batch_size = config['hparams']['batch_size']
        print("Initialized")
        self.csv_information = []
        self.test_index = 0
        self.scores_dict = {}
        self.target_names = config['hparams']['target_names']
        self.unormalize = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        images, labels = batch
        # with autocast(enabled=True):
        noise = torch.randn((images.shape[0], images.shape[1], images.shape[2], images.shape[3])).to(images.device)
        timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (images.shape[0],)).to(images.device)
        noise_pred = self.inferer(inputs=images, diffusion_model=self.model, noise=noise, timesteps=timesteps, conditioning=labels)

        train_loss = self.criterion(noise_pred.float(), noise.float())
        self.log("training_loss", train_loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # self.batches.append(batch)
        
        return train_loss
    

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images,labels = batch
        # with autocast(enabled=True):
        noise = torch.randn((images.shape[0], images.shape[1], images.shape[2], images.shape[3]), device=images.device)
        timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device).long()
        noise_pred = self.inferer(inputs=images, diffusion_model=self.model, noise=noise, timesteps=timesteps, conditioning=labels)

        val_loss = self.criterion(noise_pred.float(), noise.float())
        self.log("val_loss", val_loss, prog_bar=True, on_step=True, on_epoch=True)
        return val_loss

    
    def on_validation_epoch_end(self):

        
        labels = torch.arange(self.num_classes).to(self.device)
        current_epoch = self.current_epoch + 1
        self.noise = self.noise.to(self.device)
        if current_epoch % self.config['hparams']['validation_sample_inspect_epoch'] == 0:
            print(f'On validation epoch:{self.current_epoch} end\n')
            
            self.scheduler_DDPM.set_timesteps(num_inference_steps=self.num_train_timesteps)
            images = self.inferer.sample(input_noise=self.noise, diffusion_model=self.model, scheduler=self.scheduler_DDPM, conditioning=labels)
            images = self.unormalize(images)
            grid = make_grid(images, nrow=self.num_classes)
            self.logger.experiment.add_image(f"Generated retinal image in validation epoch end DDPM", grid, current_epoch)

            self.scheduler_DDIM.set_timesteps(num_inference_steps=self.num_inference_timesteps)
            images = self.inferer.sample(input_noise=self.noise, diffusion_model=self.model, scheduler=self.scheduler_DDIM, conditioning=labels)
            images = self.unormalize(images)
            grid = make_grid(images, nrow=self.num_classes)
            self.logger.experiment.add_image(f"Generated retinal image in validation epoch end DDIM", grid, current_epoch)

        
    
    def test_step(self,batch,batch_idx):
        accuracy = []
        timestep_list = []
        filepath_labels = self.config['exp']['file_path_labels'][batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
        # filepaths_labels = self.dm.dataloader.dataset.imgs
        filename_list = [x[0] for x in filepath_labels]
        classes = self.classes.to(self.device)
        image_list = batch[0]
        label_list = batch[1]
        
        for i in range(0, len(filename_list), self.batch_size):
        # for test_image, test_label,filename in zip(image_list, label_list,filenames):
            filenames = filename_list[self.batch_size*i: self.batch_size*(i+1)]
            testlabels = label_list[self.batch_size*i: self.batch_size*(i+1)]
            images = image_list[self.batch_size*i: self.batch_size*(i+1)]
           
            self.scores_dict.update({filename: {} for filename in filenames})

            
            for test_count,filename in enumerate(filenames):
                
                self.scores_dict[filename]['test_label'] = testlabels[test_count].item()
                test_count += 1
                self.scores_dict[filename]['class_errors_each_trial'] = []
                self.scores_dict[filename]['timestep'] = []

            

            self.test_index += 1
            test_image = images.to(self.device)
            test_image = images.repeat(self.num_classes,1,1,1)
            error = [0] * self.num_classes * len(filenames)
            

            for r in range(self.runs):
                timesteps = torch.randint(0, self.scheduler.num_train_timesteps,(1,),device=self.device).long()

                for _, filename in enumerate(filenames):
                    self.scores_dict[filename]['timestep'].append(timesteps.item())

                timesteps=torch.repeat_interleave(timesteps,self.num_classes * len(filenames),dim=0)
                
                noise = torch.randn((1, self.in_channels , self.image_h, self.image_w)).to(self.device)
                noise = torch.repeat_interleave(noise,len(filenames) * self.num_classes,dim=0)
               
                
                conditions = torch.repeat_interleave(classes, len(filenames), dim=0)
                output = self.inferer(inputs=test_image, diffusion_model=self.model, noise=noise, timesteps=timesteps, conditioning=conditions)
                losses = self.criterion(noise, output,reduction='none').mean(dim=(1,2,3)).view(-1).to(self.device)
                error = losses.cpu().numpy()
                
                error_lists = [error[i::len(filenames)] for i in range(len(filenames))]
                error_count = 0
                for filename in filenames:
                    if error_count < len(error_lists):
                        self.scores_dict[filename]['class_errors_each_trial'].append(error_lists[error_count])
                        error_count += 1

                
            for filename in filenames:

                np_class_errors = np.array(self.scores_dict[filename]['class_errors_each_trial'])
                mean_error_classes = np.mean(np_class_errors, axis=0)
                min_error_index = np.argmin(mean_error_classes, axis=0) 
                self.scores_dict[filename]['predicted_label'] = min_error_index.item()
                accuracy.append((min_error_index == self.scores_dict[filename]['test_label'])* 1.0)
                # self.class_acc.append([min_error_index, self.scores_dict[filename]['test_label']])
                csv_info = [filename, min_error_index.item(), self.scores_dict[filename]['test_label']]
                csv_info.extend(mean_error_classes.tolist())
                self.csv_information.append(csv_info)
            
        
        accuracy = torch.tensor(accuracy)
        classification_acc = torch.mean(accuracy)
        self.test_outputs.append(classification_acc)


    def on_test_epoch_end(self):
        columns = ["Image", "Predicted Label", "Test Label"]
        columns.extend(self.target_names)

        df = pd.DataFrame(self.csv_information, columns=columns)
        df.to_csv(f"{self.config['exp']['csv_dir']}/Test_trial_{self.runs}_seed_{self.seed}.csv",index=False)
        with open(f"{self.config['exp']['csv_dir']}/Test_trial_{self.runs}_seed_{self.seed}.pkl", 'wb') as pickle_file:
            pickle.dump(self.scores_dict, pickle_file)
        class_score = torch.tensor(self.test_outputs)
        score = torch.mean(class_score)
        print(f"Classification score : {score.item()}%")
        self.log("Test acc (subset accuracy)", score)


        
        self.test_outputs.clear()
        predictions = []
        labels = []
        for name, _ in self.scores_dict.items():
            predictions.append(self.scores_dict[name]['test_label'])
            labels.append(self.scores_dict[name]['predicted_label'])

        # target_names = ['AMD','Cataract','DR','Glaucoma','Myopia','Normal']
        target_names = self.target_names
        per_classes_acc = {name: accuracy_score(np.array(labels) == i, np.array(predictions) == i) for i, name in enumerate(target_names)}
        self.log_dict(per_classes_acc)
        print(per_classes_acc)
        total_acc = 0
        for _, value in per_classes_acc.items():
            total_acc += value

        avg_acc = total_acc/self.num_classes
        print(f"Average accuracy: {avg_acc}")
        self.log("Test acc (average per class)", avg_acc)


    def configure_optimizers(self):
        if self.config['hparams']['lr_scheduler']:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['hparams']['cyclic_lr']['base_lr'])
            lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer=optimizer, base_lr = self.config['hparams']['cyclic_lr']['base_lr'], max_lr=self.config['hparams']['cyclic_lr']['max_lr'],
                cycle_momentum=False, mode='exp_range'
            )
            return {
                "optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": "val_loss"}}
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            return optimizer