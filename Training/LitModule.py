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
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from torchmetrics import Accuracy

from torchvision import models, transforms
from torchvision.transforms import ToPILImage

class LitModule(pl.LightningModule):
    def __init__(self, 
                num_classes, 
                name_classes, 
                lr = 1e-4, 
                weights_decay = 1e-5, 
                device = 'cuda' if torch.cuda.is_available() else 'cpu', 
                log_every_batch = 10,
                warmup_epochs = 10,
                total_epochs = 200,
                img_to_log = 5
            ):
        super().__init__()

        self.save_hyperparameters()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.criterion_train = nn.CrossEntropyLoss(label_smoothing = 0.1)
        self.criterion_val = nn.CrossEntropyLoss()

        self.log_every_batch = log_every_batch
        self.img_to_log = img_to_log

        self.num_classes = num_classes
        self.class_names = name_classes

        self.lr = lr
        self.weights_decay = weights_decay

        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

        self.train_losses = []
        self.val_losses = []

        self.validation_error = []
        self.error_matrix = []

        self.train_augment = A.Compose([
            A.Rotate(limit=5, p=0.7),
            A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3,5), sigma_limit=0.1, p=0.3),
            # A.GaussianNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        # Le metriche devono essere registrate come moduli per essere spostate automaticamente sul device
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes, top_k=1)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes, top_k=1)

    def forward(self, X):
        return self.model(X)
    
    def on_train_epoch_start(self):
        self.train_accuracy.reset()

        e = self.current_epoch

        for name, param in self.model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

        if e == 30:
            for name, param in self.model.named_parameters():
                if name.startswith('layer4') and 'bn' not in name and 'downsample' not in name:
                    param.requires_grad = True

        if e == 60:
            for name, param in self.model.named_parameters():
                if name.startswith('layer3') and 'bn' not in name and 'downsample' not in name:
                    param.requires_grad = True

    
    def on_validation_epoch_start(self):
        self.val_accuracy.reset()

    def apply_augment(self, images):
        augmented_images = []
        batch_size = images.shape[0]
        
        for i in range(batch_size):
            img = images[i].cpu().numpy()
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)  # Permute from [C, H, W] to [H, W, C]
            
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            
            augmented = self.train_augment(image=img)
            augmented_img = augmented['image']
            augmented_images.append(augmented_img)
            
        return torch.stack(augmented_images).to(images.device)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        data, labels = data.to(self.device), labels.to(self.device)
        if data.shape[1] != 3:
            data = data.permute(0, 3, 1, 2)
        # data = self.apply_augment(data).to(self.device, dtype=torch.float32)

        logits = self(data)
        loss = self.criterion_train(logits, labels)

        self.train_losses.append(loss.item())   
        self.train_accuracy.update(logits, labels)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, labels = batch
        if data.shape[1] != 3:
            data = data.permute(0, 3, 1, 2)  # Permute from [B, H, W, C] to [B, C, H, W]
            
        data, labels = data.to(self.device), labels.to(self.device)
        logits = self(data)
        loss = self.criterion_val(logits, labels)

        self.val_losses.append(loss.item())
        self.val_accuracy.update(logits, labels)

        prob = F.softmax(logits, dim=1)
        pred_labels = torch.argmax(prob, dim=1)

        wrong_mask = (pred_labels != labels)
        wrong_idx = torch.where(wrong_mask)[0]
        self.validation_error.extend(labels[wrong_idx].cpu().numpy().tolist())

        if batch_idx % self.log_every_batch:
            prob = F.softmax(logits, dim=1)
            pred_labels = torch.argmax(prob, dim=1)

            self.log_prediction(
                data, 
                labels, 
                pred_labels, 
                self.img_to_log,  
                phase = 'Train'
            )

    def on_train_epoch_end(self) -> None:
        avg_loss = np.mean(self.train_losses) if self.train_losses else 0.0
        log_dict = {
            'avg_train_loss' : avg_loss,
            'top1_train' : self.train_accuracy.compute() * 100
        }
        self.train_losses.clear()

        self.log_dict(log_dict, prog_bar=True, on_epoch=True, on_step=False, logger=True)
    
    def on_validation_epoch_end(self) -> None:
        avg_loss = np.mean(self.val_losses) if self.val_losses else 0.0
        log_dict = {
            'avg_val_loss' : avg_loss,
            'top1_val' : self.val_accuracy.compute() * 100
        }
        self.val_losses.clear()
        self.error_matrix.append(self.validation_error)
        self.validation_error.clear()

        self.log_dict(log_dict, prog_bar=True, on_epoch=True, on_step=False, logger=True)
        
    def log_prediction(self,
            data, 
            true_labels,  # Corretto typo: true_labesl → true_labels
            pred_labels, 
            num_img_to_log, 
            phase
        ):

        # Convert tensors to numpy if needed
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu()
        if isinstance(true_labels, torch.Tensor):  # Corretto nome
            true_labels = true_labels.detach().cpu()
        if isinstance(pred_labels, torch.Tensor):
            pred_labels = pred_labels.detach().cpu()

        if len(pred_labels.shape) > 1:
            pred_labels = torch.argmax(pred_labels, dim=1)

        # Limit number of image to log
        num_img_to_log = min(num_img_to_log, data.shape[0])
        wandb_imgs = []

        for i in range(num_img_to_log):
            # Get single image and labels
            img = data[i]
            true_label = true_labels[i].item()  # CORRETTO: usa .item() per estrarre il valore scalare
            pred_label = pred_labels[i].item()  # CORRETTO: usa .item() per estrarre il valore scalare

            # Convert image tensor to PIL Image
            if img.shape[0] == 3:
                img_pil = ToPILImage()(img)
            else: 
                img = img.permute(2, 0, 1)  # Corretto typo: premute → permute
                img_pil = ToPILImage()(img)
            
            # Create caption with true and predicted labels
            true_class = self.class_names[true_label] if true_label < len(self.class_names) else f'True Classes: {true_label}'
            pred_class = self.class_names[pred_label] if pred_label < len(self.class_names) else f'Predicted Classes: {pred_label}'

            is_correct = 'YES' if true_label == pred_label else 'NO'
            caption = f'{is_correct} True: {true_class} | Pred: {pred_class}'

            # Create wandb image object
            wandb_img = wb.Image(img_pil, caption)
            wandb_imgs.append(wandb_img)
        
        # Log to wandb
        log_dict = {f'{phase}_predictions': wandb_imgs}  # Corretto nome variabile
        if hasattr(self.logger, 'experiment'):
            self.logger.experiment.log(log_dict)  # Corretto modo di loggare
    
    def configure_optimizers(self):
        cosine_epochs = self.total_epochs - self.warmup_epochs
        fc_params = list(self.model.fc.parameters())
        l4_params = list(self.model.layer4.parameters())
        l3_params = list(self.model.layer3.parameters())

        optimizer = AdamW(
            [
                {'params': fc_params, 'lr': self.lr, 'weight_decay': self.weights_decay},
                {'params': l4_params, 'lr': self.lr * 0.5, 'weight_decay': self.weights_decay},
                {'params': l3_params, 'lr': self.lr * 0.5, 'weight_decay': self.weights_decay},
            ],
            weight_decay=self.weights_decay, betas=(0.9, 0.999)
        )

        lambda_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / self.warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=1e-6
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[lambda_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

    def get_error_matrix(self):
        return  np.array(self.error_matrix)
