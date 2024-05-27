import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import torch
from utils import get_config,  seed_everything, Retinal_Cond_Lightning
from model_architecture import  Pretrained_LightningDDPM_monai

def predict_display(test_set, timesteps, module):
    with torch.inference_mode():
        # accuracy = []
        
        noise = torch.randn((1, module.in_channels , module.image_h, module.image_w)).to(module.device)
        
        test_index = 0
        for test_image, test_label in tqdm(test_set):
            class_errors = torch.empty((6,1)).to(module.device)
            classes = torch.arange(module.num_classes).to(module.device)
            output_test_images = defaultdict(list)
            output_min_error = defaultdict(list)
            test_index += 1
            test_image = torch.unsqueeze(test_image, dim=0).to(module.device)
            
            error = torch.empty_like(class_errors)

            
            for r in timesteps:
                time = r.reshape((1,)).to(module.device).long()
                
                noisy_image = module.scheduler.add_noise(original_samples=test_image, noise=noise, timesteps=time)
                for c, conditions in enumerate(classes):
                    
                    output = module.inferer(inputs=test_image, diffusion_model=module.model, noise=noise, timesteps=time, conditioning=conditions).to(module.device)
                    output_test_images[conditions.item()].append(noisy_image)
                    error[c] = module.criterion(noise, output,reduction='none').mean(dim=(1,2,3)).view(-1, 1).to(module.device)
                
                class_errors = torch.concat([class_errors, error], dim=1)
                min_error = torch.argmin(error, dim=0, keepdim=False).item()
                output_min_error[r.item()].append(min_error)
            
                    
            print(output_min_error)
            # mean_error_classes = torch.mean(class_errors, dim=1)
            # min_error_index = torch.argmin(mean_error_classes, dim=0, keepdim=False) 
            
            # accuracy.append((min_error_index == test_label).float())

            fig, axs = plt.subplots(11, 6, figsize=(75, 75))

            plt.rc('font', size=30)
            for i, timestep in enumerate(timesteps):
                timestep = timestep.item()
                for condition, image in output_test_images.items(): 
                    image = image[i].squeeze(dim=0)
                    image = image.permute(1, 2, 0).cpu()
                    axs[i, condition].imshow(image)
                    axs[i, condition].set_title(f"Condition: {condition}, Timestep: {timestep}")
                

                    if condition == 5:
                        axs[i, condition].text(1.5, 1, f"Actual class: {test_label}, Predicted class: {output_min_error[timestep][0]}", transform=axs[i, condition].transAxes)

            fig.savefig(f'subplots/condition_{test_label}_test_{test_index}_image.png')



def main():
    config = get_config("/home/ahmad/ahmad_experiments/retinal_cond_diff/conf.yaml")
    seed_everything(config["hparams"]["seed"])

    dm = Retinal_Cond_Lightning(
        config=config
    )

    dm.prepare_data()
    dm.setup(stage="test")
    module_lightning = Pretrained_LightningDDPM_monai.load_from_checkpoint(config['exp']['model_ckpt_path'], strict=False, config=config)
    t = torch.tensor([0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900],dtype=torch.float)

    test_set = dm.test

    amd = []
    cataract = []
    dr = []
    glaucoma = []
    myopia = []
    normal = []
    count_amd, count_cataract, count_dr, count_glaucoma, count_myopia, count_normal = 0,0,0,0,0,0
    amount_of_images = 5

    for image, label in test_set:
        if label == 0 and count_amd != amount_of_images:
            amd.append([image, label])
            count_amd += 1
        elif label == 1 and count_cataract != amount_of_images:
            cataract.append([image, label])
            count_cataract += 1
        elif label == 2 and count_dr != amount_of_images:
            dr.append([image, label])
            count_dr += 1
        elif label == 3 and count_glaucoma != amount_of_images:
            glaucoma.append([image, label])
            count_glaucoma += 1
        elif label == 4 and count_myopia != amount_of_images:
            myopia.append([image, label])
            count_myopia += 1
        elif label == 5 and count_normal != amount_of_images:
            normal.append([image, label])
            count_normal += 1
    

    predict_display(amd, timesteps=t, module=module_lightning)
    predict_display(cataract, timesteps=t, module=module_lightning)
    predict_display(dr, timesteps=t, module=module_lightning)
    predict_display(glaucoma, timesteps=t, module=module_lightning)
    predict_display(myopia, timesteps=t, module=module_lightning)
    predict_display(normal, timesteps=t, module=module_lightning)



if __name__=="__main__":
    main()