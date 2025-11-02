import glob
import os
import torch
import numpy as np
import pandas as pd
from skimage import exposure

from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset, random_split, Subset, WeightedRandomSampler
from typing import List, Optional, Dict
from torchvision.io import read_image
from torchvision.transforms import RandomHorizontalFlip, ColorJitter, Normalize, ToTensor, Compose, Resize
from PIL import Image 
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class ScaleToMinusOneOne:
    """Scales tensor pixel values from [0, 1] to [-1, 1]."""
    def __call__(self, tensor):
        return (tensor * 2) - 1

    def __repr__(self):
        return f"{self.__class__.__name__}()"
    

def unnormalize_minus_one_to_one(tensor):
    """Convert [-1, 1] range tensor back to [0, 1] for visualization."""
    tensor = (tensor + 1) / 2
    return torch.clamp(tensor, 0, 1)


class BlackStripCropping:
    def __init__(self, prob=0.5):
        """
        Returns tensor with black-strip cropping
        prob (float): Probability of not cropping
        percentage (float): Amount to crop
        """
        self.prob = prob

    def __call__(self, tensor):
        if np.random.random() > self.prob:
            return tensor
        temp = tensor.clone()
        _, H, _ = temp.size()
        h_portion = np.random.randint(low=14,high=28)
        top_percent = h_portion
        bottom_percent = H - h_portion

        temp[:, :top_percent, :] = 0
        temp[:, bottom_percent:, :] = 0

        return temp

    def __repr__(self):
        return f"{self.__class__.name__}(prob={self.prob}, percentage={self.percentage})"
    

def build_transform(is_train, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    # train transform
    if is_train:
        t = []
        t.append(RandomHorizontalFlip(p=0.5))
        # t.append(ColorJitter(brightness=0.3, contrast=0.2))
        t.append(ToTensor())
        t.append(BlackStripCropping(prob=0.5))
        t.append(Normalize(mean=mean, std=std))
        return Compose(t)
    # everything else
    else:
        t = []
        t.append(Resize(size=(224,224)))
        t.append(ToTensor())
        t.append(Normalize(mean=mean, std=std))
        return Compose(t)
    

    

def build_slo_transform(is_train, mean=(0.5), std=(0.5)):
    # train transform
    if is_train:
        t = []
        t.append(ToTensor())
        t.append(Resize(size=(224,224)))
        t.append(RandomHorizontalFlip(p=0.5))
        t.append(Normalize(mean=mean, std=std))
        return Compose(t)
    # everything else
    else:
        t = []
        t.append(ToTensor())
        t.append(Resize(size=(224,224)))
        t.append(Normalize(mean=mean, std=std))
        return Compose(t)


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
    def __init__(self, sample_list, transform=Compose([ToTensor()])):
        super().__init__()
        self.sample_list = sample_list
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
        t = []
        t.append(ToTensor())
        # t.append(transforms.Normalize(mean, std))
        self.transform = Compose(t)
        
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, index):
        file_path = self.sample_list[index]
        sample = Image.open(file_path)
        sample = sample.convert("RGB")
        sample = self.transform(sample)
        return sample
    
    

class SLO_Dataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []
        self.class_to_idx = {}

        class_names = sorted(entry.name for entry in os.scandir(root_dir) if entry.is_dir())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

        for cls_name in class_names:
            cls_dir = os.path.join(root_dir, cls_name)
            for file_name in os.listdir(cls_dir):
                if file_name.endswith(".npy"):
                    file_path = os.path.join(cls_dir, file_name)
                    self.samples.append((file_path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        np_array = np.load(file_path)
        np_array = exposure.equalize_adapthist(np_array)
        np_array = (np_array * 255).astype(np.uint8)
        np_array = np.transpose(np_array, (1, 2, 0))  # HWC to CHW
        # tensor = torch.from_numpy(np_array).float()

        # if tensor.dim() == 3 and tensor.shape[0] == 1:
        #     tensor = tensor.repeat(3, 1, 1)

        if self.transform:
            tensor = self.transform(np_array)

        return tensor, label


class Retinal_Predict_dataset(Dataset):
    def __init__(self, config, transform_val_test):
        self.config = config
        self.data_dir = self.config['exp']['predict_dir']
        self.size = self.config['hparams']['batch_size']
        self.num_classes = self.config['hparams']['num_classes']
        self.transform = transform_val_test
        self.sample_lists = sorted(os.listdir(self.data_dir))
        self.file_paths = [os.path.join(self.data_dir, file) for file in self.sample_lists]
        
    def __len__(self):
        return len(self.sample_lists)
    
    def __getitem__(self, index):
        file_path = self.file_paths[index]
        sample = Image.open(file_path)
        sample = sample.convert("RGB")
        
        sample = self.transform(sample)
        return sample
    
        

class FakeData_lightning(LightningDataModule):
    def __init__(self, config, size=4, image_size=[3,224,224], num_classes=0):
        super().__init__()
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
        self.config = config
        self.num_workers = len(os.sched_getaffinity(0))

    def setup(self, stage):
        self.train = Fake_Dataset(size=2*self.size, image_size=self.image_size)
        self.val = Fake_Dataset(size=self.size, image_size=self.image_size)
        self.test = Fake_Dataset(size=self.size, image_size=self.image_size)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.size, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.size, num_workers=self.num_workers)
    
class FB_BRSET_Retinal_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, config, transform=Compose([ToTensor()])):
        super().__init__()
        self.dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.map = {
            "amd": 0,
            "diabetic_retinopathy": 1,
            "myopic_fundus": 2,
            "no_disease": 3
        }
        self.config = config
        self.sensitive_attr = config['hparams']['resampling']["sensitive_attribute"]
        # self.sens_classes = config['hparams']['resampling']['sens_classes']

        self.sex_mapping = {'Male': 0, 'Female': 1, 1: 0, 2: 1}
        self.camera_mapping = {'Canon CR': 0, 'NIKON NF5050': 1}

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        label = self.dataframe.loc[index, "label"]
        img_filename = ".".join([self.dataframe.loc[index, "image_id"], 'png'])
        img_path = os.path.join(self.root_dir, label, img_filename)

        image = Image.open(img_path).convert("RGB")
        y_label = torch.tensor(self.map[label])

        # --- Handle sensitive attributes (single or multiple) ---
        attrs = self.sensitive_attr if isinstance(self.sensitive_attr, list) else [self.sensitive_attr]
        sens_values = []
        for attr in attrs:
            value = self.dataframe.loc[index, attr]
            if attr == 'patient_sex':
                value = self.sex_mapping.get(value, value)
            elif attr == 'camera':
                value = self.camera_mapping.get(value, value)
            elif attr == 'patient_age':
                value = 1 if value >= 65 else 0
            sens_values.append(value)

        # If multiple sensitive attributes, combine them into a tuple
        y_subgroup_sensitive = torch.tensor(sens_values, dtype=torch.long) if len(sens_values) > 1 else torch.tensor(sens_values[0])

        if self.transform:
            image = self.transform(image)

        return image, y_label, y_subgroup_sensitive

    def group_counts(self, resample_which='group'):
        """
        Handles resampling groups based on one or more sensitive attributes.
        """
        attrs = self.sensitive_attr if isinstance(self.sensitive_attr, list) else [self.sensitive_attr]
        df = self.dataframe.copy()

        # Convert all attributes into numeric codes
        for attr in attrs:
            if attr == 'patient_sex':
                df[attr] = df[attr].map(self.sex_mapping)
            elif attr == 'camera':
                df[attr] = df[attr].map(self.camera_mapping)
            elif attr == 'patient_age':
                df[attr] = (df[attr] >= 65).astype(int)

        # Create a unique intersectional group id
        group_array = df[attrs].astype(str).agg('_'.join, axis=1)
        unique_groups = sorted(group_array.unique())
        group_to_idx = {g: i for i, g in enumerate(unique_groups)}
        group_idx = group_array.map(group_to_idx).values

        group_tensor = torch.LongTensor(group_idx)
        group_counts = torch.bincount(group_tensor, minlength=len(unique_groups)).float()

        self._group_array = group_tensor
        self._group_counts = group_counts

        return group_idx, group_counts

    def get_weights(self, resample_which='group'):
        """
        Compute inverse frequency weights for sampling based on intersectional groups.
        """
        sens_attr, group_num = self.group_counts(resample_which)
        group_weights = [1 / x.item() if x.item() > 0 else 0 for x in group_num]
        sample_weights = [group_weights[i] for i in sens_attr]
        return sample_weights




class Retinal_Cond_Lightning_Split(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.config['exp']['num_workers'] = len(os.sched_getaffinity(0))
        self.data_dir = self.config['exp']['data_dir']
        self.size = self.config['hparams']['batch_size']
        self.num_classes = self.config['hparams']['num_classes']
        self.transform_train = build_transform(is_train=self.config['hparams']['train_is_train'])
        self.transform_val_test = build_transform(is_train=self.config['hparams']['test_val_is_train'])
        self.train_dir = os.path.join(self.data_dir,'train')
        self.val_dir = os.path.join(self.data_dir, 'val')
        self.test_dir = os.path.join(self.data_dir, 'test')

    def setup(self, stage):
        if stage == 'predict':
            predict_dataset = Retinal_Predict_dataset(self.config, transform_val_test = self.transform_val_test)
            self.predict_set = predict_dataset
            print(f"Predict dataset length: {len(self.predict_set)}")
        else:
            if stage == 'fit':
                train_dataset = ImageFolder(root=self.train_dir, transform=self.transform_train)
                valid_dataset = ImageFolder(root=self.val_dir, transform=self.transform_val_test)
                self.train = train_dataset
                self.val = valid_dataset
                print(f"Train, val length:  {len(self.train), len(self.val)}")
            elif stage == 'test':
                test_dataset = ImageFolder(root=self.test_dir, transform=self.transform_val_test)
                self.test = test_dataset
                print(f"Test length: {len(self.test)}")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.size, num_workers=self.config['exp']['num_workers'], shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.size, num_workers=self.config['exp']['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.size, num_workers=self.config['exp']['num_workers'])
    
    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=self.size, num_workers=self.config['exp']['num_workers'])
    
class Camera_Lightning_Split(Retinal_Cond_Lightning_Split):
    def __init__(self, config):
        super().__init__(config)
        self.transform_train = build_transform(is_train=self.config['hparams']['train_is_train'], mean = IMAGENET_DEFAULT_MEAN, std = IMAGENET_DEFAULT_STD)
        self.transform_val_test = build_transform(is_train=self.config['hparams']['test_val_is_train'], mean = IMAGENET_DEFAULT_MEAN, std = IMAGENET_DEFAULT_STD)
    
class SLO_Lightning_Split(Retinal_Cond_Lightning_Split):
    def __init__(self, config):
        super().__init__(config)
        self.config = config 
        self.in_channel = self.config['hparams']['DiffusionModelUnet']['in_channels']
        if self.in_channel == 1:
            mean = (0.5)
            std = (0.5)
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)   
        self.transform_train = build_slo_transform(is_train=self.config['hparams']['train_is_train'], mean=mean, std=std)
        self.transform_val_test = build_slo_transform(is_train=self.config['hparams']['test_val_is_train'], mean=mean, std=std)


    def setup(self, stage):
        if stage == 'predict':
            predict_dataset = Retinal_Predict_dataset(self.config)
            self.predict_set = predict_dataset
            print(f"Predict dataset length: {len(self.predict_set)}")
        else:
            if stage == 'fit':
                train_dataset = SLO_Dataset(root_dir=self.train_dir, transform=self.transform_train)
                valid_dataset = SLO_Dataset(root_dir=self.val_dir, transform=self.transform_val_test)
                self.train = train_dataset
                self.val = valid_dataset
                print(f"Train, val length:  {len(self.train), len(self.val)}")
            elif stage == 'test':
                test_dataset = SLO_Dataset(root_dir=self.test_dir, transform=self.transform_val_test)
                self.test = test_dataset
                print(f"Test length: {len(self.test)}")


class Resampling_Lightning_Split(Retinal_Cond_Lightning_Split):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.train_csv = os.path.join(self.config['exp']["csv_data_split_dir"],"BRSET_train_split.csv")
        self.val_csv = os.path.join(self.config['exp']["csv_data_split_dir"],"BRSET_val_split.csv")
        self.test_csv = os.path.join(self.config['exp']["csv_data_split_dir"],"BRSET_test_split.csv")

    def setup(self, stage):
        if stage == 'predict':
            predict_dataset = Retinal_Predict_dataset(self.config)
            self.predict_set = predict_dataset
            print(f"Predict dataset length: {len(self.predict_set)}")
        else:
            if stage == 'fit':
                

                train_dataset = FB_BRSET_Retinal_Dataset(csv_file=self.train_csv,
                                                         root_dir=self.train_dir, 
                                                         config=self.config, 
                                                         transform=self.transform_train)
                valid_dataset = ImageFolder(root=self.val_dir, transform=self.transform_val_test)
                self.train = train_dataset
                self.val = valid_dataset
                print(f"Train, val length:  {len(self.train), len(self.val)}")
                weights = train_dataset.get_weights(resample_which = self.config['hparams']['resampling']['resampling_which'])
                g = torch.Generator()
                g.manual_seed(self.config['hparams']['seed'])
                self.sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator = g)
            elif stage == 'test':
                test_dataset = ImageFolder(root=self.test_dir, transform=self.transform_val_test)
                self.test = test_dataset
                print(f"Test length: {len(self.test)}")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.size, num_workers=self.config['exp']['num_workers'], shuffle=False, sampler=self.sampler)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.size, num_workers=self.config['exp']['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.size, num_workers=self.config['exp']['num_workers'])
    
    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=self.size, num_workers=self.config['exp']['num_workers'])


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
        self.transform = Compose([ToTensor()])
        self.data_dir = self.config['exp']['data_dir']
        self.file_pattern = f"*.{self.config['exp']['image_extension']}"
        self.files_path = os.path.join(self.data_dir, self.file_pattern)
        self.sample_list = glob.glob(self.files_path)
        self.batch_size = self.config['hparams']['batch_size']

    def setup(self, stage):
        
        full_dataset = UK_biobank_retinal(self.sample_list, self.transform)
        dataset_size = len(full_dataset)
        train_val_set_size = int(dataset_size * 0.95)
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
   
