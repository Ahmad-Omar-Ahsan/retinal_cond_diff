import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import torch
from utils import get_config,  seed_everything, Retinal_Cond_Lightning
from model_architecture import  Pretrained_LightningDDPM_monai
from torchvision.utils import save_image
import pandas as pd
import copy


def predict(test_set, module):
    with torch.inference_mode():
        accuracy = []
        class_errors = []
        classes = module.classes.to(module.device)
        class_acc = []
        test_outputs = []

        for test_image, test_label in tqdm(test_set):
            
            test_image = torch.unsqueeze(test_image, dim=0).to(module.device)
            test_image = torch.repeat_interleave(test_image, module.batch_size
                                                    , dim=0)
            # test_image_allclass = test_image_allclass
            
            # error = np.arange(self.num_classes) * 0
            error = [0,0,0,0,0,0]
            

            for r in range(module.runs):
                timesteps = torch.randint(0, module.scheduler.num_train_timesteps,(1,),device=module.device).long()
                timesteps=torch.repeat_interleave(timesteps,module.batch_size,dim=0)

                noise = torch.randn((1, module.in_channels , module.image_h, module.image_w)).to(module.device)
                noise = torch.repeat_interleave(noise,module.batch_size,dim=0)
                
                for c in range(0,len(classes), module.batch_size):
                    conditions = classes[c: c+module.batch_size]
                    output = module.inferer(inputs=test_image, diffusion_model=module.model, noise=noise, timesteps=timesteps, conditioning=conditions)
                    value = module.criterion(noise, output,reduction='none').mean(dim=(1,2,3)).view(-1).to(module.device)
                    error[c: c+module.batch_size] = value.cpu().numpy()
                
                class_errors.append(copy.deepcopy(error))
        
            
            np_class_errors = np.array(class_errors)
            mean_error_classes = np.mean(np_class_errors, axis=0)
            min_error_index = np.argmin(mean_error_classes, axis=0) 
        
            accuracy.append((min_error_index == test_label).astype('float'))
            class_acc.append([min_error_index, test_label])
        # accuracy = torch.mean(accuracy)
        # accuracy = torch.any()
        accuracy = torch.tensor(accuracy)
        classification_acc = torch.mean(accuracy)
        # self.log("Test accuracy", classification_acc)
        # self.test_outputs[dataloader_idx].append({"preds": 100*classification_acc})
        test_outputs.append(100*classification_acc)


        class_score = torch.tensor(test_outputs)
        score = torch.mean(class_score)
        print(f"Classification score : {score.item()}%")
        

        mapping = {
            '0' : "AMD",
            '1': "Cataract",
            '2': "DR",
            '3' : "Myopia",
            '4' : "Glaucoma",
            '5' : "Normal"
        }
        
        class_scores = {}
        label_count = {}
        for scores in class_acc:
            label = scores[1]
            prediction = scores[0]
            if label == prediction:
                class_scores[str(label)] = class_scores.get(str(label), 0) + 1
            label_count[str(label)] = label_count.get(str(label),0) + 1
        
        class_scores = dict(sorted(class_scores.items()))
        label_count = dict(sorted(label_count.items()))
        class_acc_dict = {}
        print(class_scores, label_count)
        for key1, key2 in zip(class_scores, label_count):
            correct = class_scores[key1]
            count = label_count[key2]

            class_accuracy = 100 * (correct / count)
            class_acc_dict[key1] = class_accuracy
            print(f"For {mapping[key1]} accuracy is: {class_accuracy}")
        

def predict_display(test_set, timesteps, module, test_index_list):
    with torch.inference_mode():
        # accuracy = []
        
        test_index = 0
        for test_image, test_label in tqdm(test_set):
            test_index += 1
            if test_index not in test_index_list:
                # print(test_index)
                # print('here')
                continue
            class_errors = []
            
            classes = torch.arange(module.num_classes).to(module.device)
            output_test_images = defaultdict(list)
            output_min_error = defaultdict(list)
            
            test_image = torch.unsqueeze(test_image, dim=0).to(module.device)
            
            error = [0,0,0,0,0,0]
            
            noise = torch.randn((1, module.in_channels , module.image_h, module.image_w)).to(module.device)
            for r in timesteps:
                time = r.reshape((1,)).to(module.device).long()
                
                noisy_image = module.scheduler.add_noise(original_samples=test_image, noise=noise, timesteps=time)
                for c, conditions in enumerate(classes):
                    
                    output = module.inferer(inputs=test_image, diffusion_model=module.model, noise=noise, timesteps=time, conditioning=conditions).to(module.device)
                    output_test_images[conditions.item()].append(noisy_image)
                    value = module.criterion(noise, output,reduction='none').mean(dim=(1,2,3)).view(-1, 1).to(module.device)
                    error[c] = value[0].item()
                class_errors.append(copy.deepcopy(error))
                
                min_error = torch.argmin(torch.tensor(error), dim=0, keepdim=False).item()
                output_min_error[r.item()].append(min_error)
            
            class_errors = torch.tensor(class_errors)
            mean_class_error = torch.mean(class_errors,dim=0)
            min_indexes = torch.argmin(mean_class_error)
            # class_errors = torch.concat(class_errors, dim=1)
            
            # save_image(test_image,fp=f'test_images/label_{test_label}_{test_index}.png')     
            print(f"For test index: {test_index} and test label: {test_label}")
            # print(class_errors)
            df = pd.DataFrame(class_errors.cpu().numpy())
            df.index = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900]
            df.columns = ["AMD", "Cataract", "DR", "Glaucoma", "Myopia", "Normal"]
            print(df)
            print(f"Index for each timestep: {output_min_error}")
            print(f"Mean class error: {mean_class_error}")
            print(f"Predicted class: {min_indexes.item()}")
            # mean_error_classes = torch.mean(class_errors, dim=1)
            # min_error_index = torch.argmin(mean_error_classes, dim=0, keepdim=False) 
            
            # accuracy.append((min_error_index == test_label).float())

            # fig, axs = plt.subplots(11, 6, figsize=(75, 75))

            # plt.rc('font', size=30)
            # for i, timestep in enumerate(timesteps):
            #     timestep = timestep.item()
            #     for condition, image in output_test_images.items(): 
            #         image = image[i].squeeze(dim=0)
            #         image = image.permute(1, 2, 0).cpu()
            #         axs[i, condition].imshow(image)
            #         axs[i, condition].set_title(f"Condition: {condition}, Timestep: {timestep}")
                

            #         if condition == 5:
            #             axs[i, condition].text(1.5, 1, f"Actual class: {test_label}, Predicted class: {output_min_error[timestep][0]}", transform=axs[i, condition].transAxes)

            # fig.savefig(f'subplots/condition_{test_label}_test_{test_index}_image.png')



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

    # test_set = dm.test

    # amd = []
    # cataract = []
    # dr = []
    # glaucoma = []
    # myopia = []
    # normal = []
    # count_amd, count_cataract, count_dr, count_glaucoma, count_myopia, count_normal = 0,0,0,0,0,0
    # amount_of_images = 50

    # for image, label in test_set:
    #     if label == 0 and count_amd != amount_of_images:
    #         amd.append([image, label])
    #         count_amd += 1
    #     elif label == 1 and count_cataract != amount_of_images:
    #         cataract.append([image, label])
    #         count_cataract += 1
    #     elif label == 2 and count_dr != amount_of_images:
    #         dr.append([image, label])
    #         count_dr += 1
    #     elif label == 3 and count_glaucoma != amount_of_images:
    #         glaucoma.append([image, label])
    #         count_glaucoma += 1
    #     elif label == 4 and count_myopia != amount_of_images:
    #         myopia.append([image, label])
    #         count_myopia += 1
    #     elif label == 5 and count_normal != amount_of_images:
    #         normal.append([image, label])
    #         count_normal += 1
    

    # predict_display(amd, timesteps=t, module=module_lightning, test_index_list=[2,3])
    # predict_display(cataract, timesteps=t, module=module_lightning, test_index_list=[6])
    # predict_display(dr, timesteps=t, module=module_lightning, test_index_list=[3])
    # predict_display(glaucoma, timesteps=t, module=module_lightning, test_index_list=[2])
    # predict_display(myopia, timesteps=t, module=module_lightning, test_index_list=[7])
    # predict_display(normal, timesteps=t, module=module_lightning, test_index_list=[4])

    count_amd, count_cataract, count_dr, count_glaucoma, count_myopia, count_normal = 0,0,0,0,0,0
    amount_of_images = 10
    pseudo_train_set = []
    train_set = dm.train

    for image, label in train_set:
        if label == 0 and count_amd != amount_of_images:
            pseudo_train_set.append([image, label])
            count_amd += 1
        elif label == 1 and count_cataract != amount_of_images:
            pseudo_train_set.append([image, label])
            count_cataract += 1
        elif label == 2 and count_dr != amount_of_images:
            pseudo_train_set.append([image, label])
            count_dr += 1
        elif label == 3 and count_glaucoma != amount_of_images:
            pseudo_train_set.append([image, label])
            count_glaucoma += 1
        elif label == 4 and count_myopia != amount_of_images:
            pseudo_train_set.append([image, label])
            count_myopia += 1
        elif label == 5 and count_normal != amount_of_images:
            pseudo_train_set.append([image, label])
            count_normal += 1
            
    predict(pseudo_train_set,  module=module_lightning)
    



if __name__=="__main__":
    main()