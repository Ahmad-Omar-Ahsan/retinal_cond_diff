import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics
import torch
from sklearn.metrics import accuracy_score
import numpy as np

class MLP_classifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.lr = config['hparams']['learning_rate']
        
        self.fc1 = nn.Linear(1024, self.config['hparams']['mlp']['layer_1'])
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(self.config['hparams']['mlp']['layer_1'], self.config['hparams']['mlp']['num_classes'])

        self.num_classes = self.config['hparams']['mlp']['num_classes']
        self.criterion = F.cross_entropy

        self.net = nn.Sequential(self.fc1, self.act1, self.fc2)
        self.train_prediction_labels = []
        self.val_prediction_labels = []
        self.test_prediction_labels = []
        self.test_labels = []

        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.config['hparams']['mlp']['num_classes'])
        self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.config['hparams']['mlp']['num_classes'])
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.config['hparams']['mlp']['num_classes'])

        self.preds_labels = []
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        latents, labels = batch

        predictions = self.net(latents)
        train_loss = self.criterion(input=predictions, target=labels)

        preds = F.softmax(predictions, dim=1).argmax(dim=1)
        self.train_acc(preds, labels)

        self.log("training_loss", train_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)

        return train_loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        latents, labels = batch

        predictions = self.net(latents)
        val_loss = self.criterion(input=predictions, target=labels)
        preds = F.softmax(predictions, dim=1).argmax(dim=1)
        self.valid_acc(preds, labels)

        self.log("val_loss", val_loss, on_step=True, on_epoch=True)
        self.log('val_acc', self.valid_acc, on_step=True, on_epoch=True)


    def test_step(self, batch):
        latents, labels = batch

        predictions = self.net(latents)

        preds = F.softmax(predictions, dim=1).argmax(dim=1)
        self.test_acc(preds, labels)
        # self.log("Test_acc", self.test_acc, on_epoch=True, on_step=True)
        self.test_prediction_labels.extend(preds.cpu().numpy())
        self.test_labels.extend(labels.cpu().numpy())
        


    def on_test_epoch_end(self):

       
        target_names = ['AMD','Cataract','DR','Glaucoma','Myopia','Normal']
        per_classes_acc = {name: accuracy_score(np.array(self.test_labels) == i, np.array(self.test_prediction_labels) == i) for i, name in enumerate(target_names)}
        self.log_dict(per_classes_acc)
        total_acc = 0
        for key,value in per_classes_acc.items():
            total_acc += value
        
        avg_acc = total_acc/self.num_classes
        self.log("Accuracy_avg_class", avg_acc)
        

        


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        return optimizer