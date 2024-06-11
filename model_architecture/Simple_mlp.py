import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics
import torch

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

        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.config['hparams']['mlp']['num_classes'])
        self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.config['hparams']['mlp']['num_classes'])
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.config['hparams']['mlp']['num_classes'])

        self.preds_labels = []

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
        self.log("Test_acc", self.test_acc, on_epoch=True, on_step=True)
        
        for pred, label in zip(preds,labels):
            self.preds_labels.append([pred, label])


    def on_test_epoch_end(self):

        class_correct = {str(label): 0 for label in range(self.num_classes)}
        label_count = {str(label): 0 for label in range(self.num_classes)}
        class_acc_dict = {}
        mapping = {
            '0' : "AMD",
            '1': "Cataract",
            '2': "DR",
            '3' : "Myopia",
            '4' : "Glaucoma",
            '5' : "Normal"
        }

        for preds_labels in self.preds_labels:
            pred = preds_labels[0].item()
            label = preds_labels[1].item()

            if label == pred:
                class_correct[str(label)] = class_correct.get(str(label), 0) + 1
            label_count[str(label)] = label_count.get(str(label), 0) + 1

        class_correct = dict(sorted(class_correct.items()))
        label_count = dict(sorted(label_count.items()))
        
        print(class_correct, label_count)
        for key1, key2 in zip(class_correct, label_count):
            correct = class_correct[key1]
            count = label_count[key2]

            class_accuracy = 100 * (correct / count)
            class_acc_dict[key1] = class_accuracy
            print(f"For {mapping[key1]} accuracy is: {class_accuracy}")
        self.log_dict(class_acc_dict)

        self.preds_labels.clear()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        return optimizer