import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics
import torch
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
import pandas as pd
from torchvision import models

class EfficientNet_B3(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model = models.efficientnet_b3(weights="IMAGENET1K_V1")
        self.model.classifier[1]= nn.Linear(in_features=1536, out_features=self.config['hparams']['num_classes'])
        
        self.criterion = F.cross_entropy

        self.train_prediction_labels = []
        self.val_prediction_labels = []
        self.test_prediction_labels = []
        self.test_labels = []

        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.config['hparams']['mlp']['num_classes'])
        self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.config['hparams']['mlp']['num_classes'])
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.config['hparams']['mlp']['num_classes'])

        self.preds_labels = []
        self.lr = self.config['hparams']['learning_rate']
        self.num_classes = self.config['hparams']['num_classes']
        self.batch_size = self.config['hparams']['batch_size']
        self.seed = self.config['hparams']['seed']
        self.save_hyperparameters()
        self.csv_information = []
        

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        images, labels = batch

        predictions = self.model(images)
        train_loss = self.criterion(input=predictions, target=labels)

        preds = F.softmax(predictions, dim=1).argmax(dim=1)
        self.train_acc(preds, labels)

        self.log("training_loss", train_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)

        return train_loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images, labels = batch

        predictions = self.model(images)
        val_loss = self.criterion(input=predictions, target=labels)
        preds = F.softmax(predictions, dim=1).argmax(dim=1)
        self.valid_acc(preds, labels)

        self.log("val_loss", val_loss, on_step=True, on_epoch=True)
        self.log('val_acc', self.valid_acc, on_step=True, on_epoch=True)


    def test_step(self, batch,batch_idx):
        images, labels = batch
        predictions = self.model(images)
        filepath_labels = self.config['exp']['file_path_labels'][batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
        filenames = [x[0] for x in filepath_labels]
        preds = F.softmax(predictions, dim=1).argmax(dim=1)
        preds_cpu = preds.cpu().numpy()
        labels_cpu = labels.cpu().numpy()
        self.test_acc(preds, labels)
        self.log("Test_acc", self.test_acc, on_epoch=True, on_step=True)
        csv_info = list(zip(filenames, labels_cpu, preds_cpu))
        self.csv_information.extend(csv_info)
        self.test_prediction_labels.extend(preds_cpu)
        self.test_labels.extend(labels_cpu)
        


    def on_test_epoch_end(self):
       
        target_names = ['AMD','Cataract','DR','Glaucoma','Myopia','Normal']
        per_classes_acc = {name: accuracy_score(np.array(self.test_labels) == i, np.array(self.test_prediction_labels) == i) for i, name in enumerate(target_names)}
        self.log_dict(per_classes_acc)
        total_acc = 0
        for key,value in per_classes_acc.items():
            total_acc += value
        
        avg_acc = total_acc/self.num_classes
        self.log("Accuracy_avg_class", avg_acc)

        columns = ["Image", "Actual Label","Predicted Label"]
        df = pd.DataFrame(self.csv_information, columns=columns)
        df.to_csv(f"{self.config['exp']['csv_dir']}/Test_seed_{self.seed}.csv",index=False)



    def predict_step(self, batch, batch_idx):
        images = batch
        filepath_labels = self.config['exp']['file_path_labels'][batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]

        predictions = self.model(images)

        preds = F.softmax(predictions, dim=1).argmax(dim=1)
        preds_cpu = preds.cpu().numpy()
        # self.test_acc(preds, labels)
        # self.log("Test_acc", self.test_acc, on_epoch=True, on_step=True)
        # self.test_prediction_labels.extend(preds_cpu)
        csv_info = list(zip(filepath_labels, preds_cpu))
        self.csv_information.extend(csv_info)
        # self.test_labels.extend(labels.cpu().numpy())

    
    def on_predict_epoch_end(self):
        columns = ["Image", "Predicted Label",]
        df = pd.DataFrame(self.csv_information, columns=columns)
        df.to_csv(f"{self.config['exp']['csv_dir']}/Predict_seed_{self.seed}.csv",index=False)
        # with open(f"{self.config['exp']['csv_dir']}/Predict_seed_{self.seed}.pkl", 'wb') as pickle_file:
        #     pickle.dump(self.scores_dict, pickle_file)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,mode='min', patience=3,
        )
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss"}}
    


class Swin_B(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model = models.swin_b(weights="IMAGENET1K_V1")
        self.num_features = self.model.head.in_features
        self.model.head = nn.Linear(self.num_features, self.config['hparams']['num_classes'])
        
        self.criterion = F.cross_entropy

        self.train_prediction_labels = []
        self.val_prediction_labels = []
        self.test_prediction_labels = []
        self.test_labels = []

        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.config['hparams']['mlp']['num_classes'])
        self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.config['hparams']['mlp']['num_classes'])
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.config['hparams']['mlp']['num_classes'])

        self.preds_labels = []
        self.lr = self.config['hparams']['learning_rate']
        self.num_classes = self.config['hparams']['num_classes']
        self.batch_size = self.config['hparams']['batch_size']
        self.seed = self.config['hparams']['seed']
        self.save_hyperparameters()
        self.csv_information = []
        

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        images, labels = batch

        predictions = self.model(images)
        train_loss = self.criterion(input=predictions, target=labels)

        preds = F.softmax(predictions, dim=1).argmax(dim=1)
        self.train_acc(preds, labels)

        self.log("training_loss", train_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)

        return train_loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images, labels = batch

        predictions = self.model(images)
        val_loss = self.criterion(input=predictions, target=labels)
        preds = F.softmax(predictions, dim=1).argmax(dim=1)
        self.valid_acc(preds, labels)

        self.log("val_loss", val_loss, on_step=True, on_epoch=True)
        self.log('val_acc', self.valid_acc, on_step=True, on_epoch=True)


    def test_step(self, batch,batch_idx):
        images, labels = batch
        predictions = self.model(images)
        filepath_labels = self.config['exp']['file_path_labels'][batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
        filenames = [x[0] for x in filepath_labels]
        preds = F.softmax(predictions, dim=1).argmax(dim=1)
        preds_cpu = preds.cpu().numpy()
        labels_cpu = labels.cpu().numpy()
        self.test_acc(preds, labels)
        self.log("Test_acc", self.test_acc, on_epoch=True, on_step=True)
        csv_info = list(zip(filenames, labels_cpu, preds_cpu))
        self.csv_information.extend(csv_info)
        self.test_prediction_labels.extend(preds_cpu)
        self.test_labels.extend(labels_cpu)
        


    def on_test_epoch_end(self):
       
        target_names = ['AMD','Cataract','DR','Glaucoma','Myopia','Normal']
        per_classes_acc = {name: accuracy_score(np.array(self.test_labels) == i, np.array(self.test_prediction_labels) == i) for i, name in enumerate(target_names)}
        self.log_dict(per_classes_acc)
        total_acc = 0
        for key,value in per_classes_acc.items():
            total_acc += value
        
        avg_acc = total_acc/self.num_classes
        self.log("Accuracy_avg_class", avg_acc)

        columns = ["Image", "Actual Label","Predicted Label"]
        df = pd.DataFrame(self.csv_information, columns=columns)
        df.to_csv(f"{self.config['exp']['csv_dir']}/Test_seed_{self.seed}.csv",index=False)



    def predict_step(self, batch, batch_idx):
        images = batch
        filepath_labels = self.config['exp']['file_path_labels'][batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]

        predictions = self.model(images)

        preds = F.softmax(predictions, dim=1).argmax(dim=1)
        preds_cpu = preds.cpu().numpy()
        # self.test_acc(preds, labels)
        # self.log("Test_acc", self.test_acc, on_epoch=True, on_step=True)
        # self.test_prediction_labels.extend(preds_cpu)
        csv_info = list(zip(filepath_labels, preds_cpu))
        self.csv_information.extend(csv_info)
        # self.test_labels.extend(labels.cpu().numpy())

    
    def on_predict_epoch_end(self):
        columns = ["Image", "Predicted Label",]
        df = pd.DataFrame(self.csv_information, columns=columns)
        df.to_csv(f"{self.config['exp']['csv_dir']}/Predict_seed_{self.seed}.csv",index=False)
        # with open(f"{self.config['exp']['csv_dir']}/Predict_seed_{self.seed}.pkl", 'wb') as pickle_file:
        #     pickle.dump(self.scores_dict, pickle_file)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,mode='min', patience=3,
        )
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss"}}