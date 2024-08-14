import glob
import os
import torch
import numpy as np
import pandas as pd

from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from typing import List, Optional, Dict
from torchvision.io import read_image
from torchvision import transforms
from PIL import Image 
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

class Fake_Dataset(Dataset):
    def __init__(self, size=4, image_size=[3,224,224]):
        super().__init__()
        self.size = size
        self.image_size = image_size
        self.images = torch.rand(size=(self.size, self.image_size[0], self.image_size[1], self.image_size[2]))

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.images[index]



class PickleDataset(Dataset):
    def __init__(self, pickle_file):
        super().__init__()
        self.data = []
        df = pd.read_pickle(pickle_file)
        self.latents = torch.tensor(df['feature'])
        self.labels = torch.tensor(df['label'])
        self.data = [self.latents, self.labels]

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        latent, label = self.data[0][idx], self.data[1][idx]
        return latent, label
    

class UK_biobank_retinal(Dataset):
    def __init__(self, sample_list, transform=transforms.Compose([transforms.ToTensor()])):
        super().__init__()
        self.sample_list = sample_list
        self.transform = transform
    
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, index):
        file_path = self.sample_list[index]
        sample = Image.open(file_path)
        sample = sample.convert("RGB")
        sample = self.transform(sample)
        return sample
    
    

class ODIR_Dataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config


class FakeData_lightning(LightningDataModule):
    def __init__(self, config, size=4, image_size=[3,224,224], num_classes=0):
        super().__init__()
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
        self.config = config

    def setup(self, stage):
        self.train = Fake_Dataset(size=2*self.size, image_size=self.image_size)
        self.val = Fake_Dataset(size=self.size, image_size=self.image_size)
        self.test = Fake_Dataset(size=self.size, image_size=self.image_size)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.size, num_workers=self.config['exp']['num_workers'])
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.size, num_workers=self.config['exp']['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.size, num_workers=self.config['exp']['num_workers'])
    
    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.size, num_workers=self.config['exp']['num_workers'])
    

class Retinal_Cond_Lightning(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = self.config['exp']['data_dir']
        self.size = self.config['hparams']['batch_size']
        self.num_classes = self.config['hparams']['num_classes']
        self.transform = transforms.Compose([transforms.Resize(224),transforms.ToTensor()
                                             ])

    def setup(self, stage):
        dataset = ImageFolder(root=self.data_dir, transform=self.transform)
        targets = dataset.targets
        train_idx, temp_idx = train_test_split(
            np.arange(len(targets)),
            test_size=0.1,
            shuffle=True,
            stratify=targets
        )
        valid_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.5,
            shuffle=True,
            stratify=np.array(targets)[temp_idx]
        )
        train_dataset = Subset(dataset, train_idx)
        valid_dataset = Subset(dataset, valid_idx)
        test_dataset = Subset(dataset, test_idx)
        self.train = train_dataset
        self.test = test_dataset
        self.val = valid_dataset
        print(f"Train, val and test length:  {len(self.train), len(self.val), len(self.test)}")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.size, num_workers=self.config['exp']['num_workers'])
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.size, num_workers=self.config['exp']['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.size, num_workers=self.config['exp']['num_workers'])
    
    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.size, num_workers=self.config['exp']['num_workers'])
    

class Retinal_Cond_Lightning_Split(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = self.config['exp']['data_dir']
        self.size = self.config['hparams']['batch_size']
        self.num_classes = self.config['hparams']['num_classes']
        self.transform = transforms.Compose([transforms.Resize(224),transforms.ToTensor()
                                             ])
        self.train_dir = os.path.join(self.data_dir,'train')
        self.val_dir = os.path.join(self.data_dir, 'val')
        self.test_dir = os.path.join(self.data_dir, 'test')

    def setup(self, stage):
        train_dataset = ImageFolder(root=self.train_dir, transform=self.transform)
        valid_dataset = ImageFolder(root=self.val_dir, transform=self.transform)
        test_dataset = ImageFolder(root=self.test_dir, transform=self.transform)
        self.train = train_dataset
        self.test = test_dataset
        self.val = valid_dataset
        print(f"Train, val and test length:  {len(self.train), len(self.val), len(self.test)}")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.size, num_workers=self.config['exp']['num_workers'], shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.size, num_workers=self.config['exp']['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.size, num_workers=self.config['exp']['num_workers'])
    
    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.size, num_workers=self.config['exp']['num_workers'])



class Pickle_Lightning(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pickle_dir = config['exp']['pickle_dir']
        self.pickle_train_file = os.path.join(self.pickle_dir, "Train_Feature_latent.pkl")
        self.pickle_test_file = os.path.join(self.pickle_dir, "Test_Feature_latent.pkl")
        self.pickle_val_file = os.path.join(self.pickle_dir, "Val_Feature_latent.pkl")

        self.size = self.config['hparams']['batch_size']
        self.num_classes = self.config['hparams']['num_classes']

    def setup(self, stage):
        train_dataset = PickleDataset(self.pickle_train_file)
        valid_dataset = PickleDataset(self.pickle_val_file)
        test_dataset= PickleDataset(self.pickle_test_file)

        self.train = train_dataset
        self.test = test_dataset
        self.val = valid_dataset
        print(f"Train, val and test length:  {len(self.train), len(self.val), len(self.test)}")


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.size, num_workers=self.config['exp']['num_workers'])
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.size, num_workers=self.config['exp']['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.size, num_workers=self.config['exp']['num_workers'])
    
    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.size, num_workers=self.config['exp']['num_workers'])




class UK_biobank_data_module(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def prepare_data(self):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.data_dir = self.config['exp']['data_dir']
        self.file_pattern = f"*.{self.config['exp']['image_extension']}"
        self.files_path = os.path.join(self.data_dir, self.file_pattern)
        self.sample_list = glob.glob(self.files_path)
        self.batch_size = self.config['hparams']['batch_size']

    def setup(self, stage):
        
        full_dataset = UK_biobank_retinal(self.sample_list, self.transform)
        dataset_size = len(full_dataset)
        train_val_set_size = int(dataset_size * 0.9)
        test_set_size = dataset_size - train_val_set_size
        self.train_val, self.test = random_split(full_dataset,[train_val_set_size, test_set_size])
        
        
        train_size = int(len(self.train_val) * 0.9)
        val_size = len(self.train_val) - train_size
        self.train, self.val = random_split(self.train_val, [train_size, val_size])

    
        self.test = self.test
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.config['exp']['num_workers'])
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.config['exp']['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.config['exp']['num_workers'])
    
    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.config['exp']['num_workers'])




if __name__=='__main__':
    config_dir = "/home/ahmad/ahmad_experiments/retinal_cond_diff/conf.yaml"
#     data_dir = "/home/ahmad/ahmad_experiments/retinal_data/Dataset_SPIE"
#     transform_image = transforms.Compose([
#     transforms.ToTensor(),
#    ])
#     dataset = ImageFolder(root=data_dir, transform=transform_image)
    # sample, label = dataset[0]
    # print(sample.shape, label)
    # print(dataset.class_to_idx)
    # config = get_config(config_file=config_dir)
   
