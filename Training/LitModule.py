from math import log
from token import tok_name
import pytorch_lightning as pl
import wandb as wb
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from torchmetrics import Accuracy

from torchvision import models, transforms
from torchvision.transforms import ToPILImage

class LitModule(pl.LightningModule):
    def __init__(self, 
                num_classes, 
                name_classes, 
                lr = 1e-4, 
                weights_decay = 1e-5, 
                device = 'cpu', 
                log_every_batch = 10,
                warmup_epochs = 5,
                total_epochs = 200
            ):
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

        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

        self.train_losses = []
        self.val_losses = []

        self.train_augment = A.Compose([
            A.Rotate(limit=5, p=0.7),
            A.RandomSizedCrop(height=224, width=224, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3,5), sigma_limit=0.1, p=0.3),
            A.GassuianNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
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

    def apply_augment(self, images):
        augmented_images = []
        batch_size = images.shape[0]
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)  # Permute from [C, H, W] to [H, W, C]
        
        if img.max <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        augmented = self.train_augment(image=img)
        augmented_img = augmented['image']

        augmented_images.append(augmented_img)
        return torch.stack(augmented_images).to(images.device)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        data, labels = data.to(self.device), labels.to(self.device)
        data = self.apply_augment(data).to(self.device, dtype=torch.float32)

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
        e = self.current_epoch

        for name, param in self.model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

        if e == 20:
            for name, param in self.model.named_parameters():
                if name.startswith('layer4') and 'bn' not in name and 'downsample' not in name:
                    param.requires_grad = True
    
    def configure_optimizers(self):
        cosine_epochs = self.total_epochs - self.warmup_epochs
        fc_params = list(self.model.fc.parameters())
        l4_params = list(self.model.layer4.parameters())

        optimizer = AdamW(
            [
                {'params': fc_params, 'lr': self.lr},
                {'params': l4_params, 'lr': self.lr * 0.05},
            ],
            weight_decay=self.weights_decay, betas=(0.9, 0.999)
        )

        linear_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=1e-6
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[linear_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

