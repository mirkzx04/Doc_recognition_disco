import os
import cv2
import numpy as np
import wandb as wb
import pytorch_lightning as pl
import torch


from pytorch_lightning.loggers import WandbLogger

dataset_path = r'\\10.5.1.36\dataset_IA\dataset_pdf_v1'

from Dataset_classes.DocDataset import DocumentDataset

from Training.LitModule import LitModule

if __name__ == "__main__":
    prject_name = ''

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'=== DEVICE : {device} ===')

    # Setting logger
    wandb_logger = WandbLogger(project=prject_name, log_model=True)

    # Load documentation dataset
    doc_dataset = DocumentDataset(size=(224, 224), blur_kernel=(4,4), dataset_path=dataset_path)
    doc_dataset.create_clean_dataset()
    doc_dataset.create_dataset()

    train_ratio, val_ratio = 0.8, 0.2
    train_set, val_set = doc_dataset.split_dataset(train_ratio, val_ratio)

    num_classes = 3
    lr = 1e-3
    weights_decay = 1e-4
    train_epochs = 300

    model = LitModule(num_classes=num_classes, lr=lr, weights_decay=weights_decay)
    trainer = pl.Trainer(
        max_epochs=train_epochs,
        logger=wandb_logger,
        accelerator='auto',
        devices=device,
        log_every_n_steps=10,
        deterministic=True,
        enable_checkpointing=True,
        gradient_clip_algorithm='norm',
        gradient_clip_val=1.0,
    )








