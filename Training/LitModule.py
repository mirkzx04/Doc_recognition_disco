from math import log
from token import tok_name
import pytorch_lightning as pl
import wandb as wb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchmetrics import Accuracy

from torchvision import models, transforms
from torchvision.transforms import ToPILImage

class LitModule(pl.LightningModule):
    def __init__(self, num_classes, name_classes, lr = 1e-4, weights_decay = 1e-5, device = 'cpu', log_every_batch = 10):
        super().__init__()

        self.save_hyperparameters()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.criterion = nn.CrossEntropyLoss()

        self.log_every_batch = log_every_batch

        self.num_classes = num_classes
        self.class_names = name_classes

        self.lr = lr
        self.weights_decay = weights_decay

        self.train_losses = []
        self.val_losses = []

        self.accuracy_metrics = {
            'top1_train' : Accuracy(task='multiclass', num_classes=num_classes, top_k=1).to(device),
            'top1_val' : Accuracy(task='multiclass', num_classes=num_classes, top_k=1).to(device),
        }

    def forward(self, X):
        return self.model(X)
    
    def on_train_epoch_start(self):
        self.accuracy_metrics['top1_train'].reset()

        e = self.current_epoch

        if e == 0:
            self.set_model_params()
    
    def on_validation_epoch_start(self):
        self.accuracy_metrics['top1_val'].reset()

    def training_step(self, batch, batch_idx):
        data, labels = batch
        logits = self(data)
        loss = self.criterion(logits, labels)

        self.train_losses.append(loss.item())   
        self.accuracy_metrics['top1_train'].update(logits, labels)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, labels = batch
        logits = self(data)
        loss = self.criterion(logits, labels)

        self.val_losses.append(loss.item())
        self.accuracy_metrics['top1_val'].update(logits, labels)

        if batch_idx % self.log_every_batch:
            prob = F.softmax(logits, dim=1)
            pred_labels = torch.argmax(prob, dim=1)

            self.log_prediction(
                batch, 
                labels, 
                pred_labels, 
                self.img_to_log,  
                phase = 'Train'
            )

    def on_train_epoch_end(self) -> None:
        log_dict = {
            'avg_train_loss' : self.train_losses.mean().item(),
            'top1_train' : self.accuracy_metrics['top1_train'].compute() * 100
        }
        self.train_losses.clear()

        self.log(log_dict, prog_bar=True, on_epoch=True, on_step=False, logger=True)
    
    def on_validation_epoch_end(self) -> None:
        log_dict = {
            'avg_val_loss' : self.val_losses.mean().item(),
            'top1_val' : self.accuracy_metrics['top1_val'].compute() * 100
        }
        self.val_losses.clear()

        self.log(log_dict, prog_bar=True, on_epoch=True, on_step=False, logger=True)
        
    def log_prediction(self,
            data, 
            true_labesl, 
            pred_labels, 
            num_img_to_log, 
            phase
        ):

        # Convert tensors to numpy if needed
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu()
        if isinstance(true_labesl, torch.Tensor):
            true_labesl  = true_labesl.detach().cpu()
        if isinstance(pred_labels, torch.Tensor):
            pred_labels = pred_labels.detach().cpu()

        if len(pred_labels.shape) > 1:
            pred_labels = torch.argmax(pred_labels, dim = 1)

        # Limit number of image to log
        num_img_to_log = min(num_img_to_log, data.shape[0])
        wandb_imgs = []

        for i in range(num_img_to_log):
            # Get singles image and lables
            img = data[i]
            true_label = data[i]
            pred_label = pred_labels[i]

            # Convert images tensor to PIL Image, handle different image formats 
            if img.shape[0] == 3:
                img_pil = ToPILImage()(img)
            else : 
                img = img.premute(2, 0, 1) # Permute from [H, W, C] to [C, H, W]
                img_pil = ToPILImage()(img)
            
            # Create caption with true and predicted lables
            true_class = self.class_names[true_label] if true_label < len(self.class_names) else f'True Classes : {true_label}'
            pred_class = self.class_names[pred_label] if true_label < len(self.class_names) else f'Predicted Classes : {pred_label}'

            is_correct = 'YES' if true_label == pred_label else 'NO'
            caption =f'{is_correct} True : {true_class} | Pred : {pred_class}'

            # Create wandb image object
            wandb_img = wb.Image(
                img_pil, 
                caption
            )
            wandb_imgs.append(wandb_img)
        
        # Log to wandb
        log_dic = {f'{phase}_predictions' : wandb_imgs}
        self.log(log_dic, on_epoch=True, on_step=True)

    def set_model_params(self):
        for name, param in self.model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    