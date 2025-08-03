import os
import cv2
import numpy as np
import wandb as wb

import torch

import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger

dataset_path = r"\\10.5.1.36\dataset_IA\dataset_pdf_Solo_Rinnovi\images"

from Dataset_classes.DocDataset import DocumentDataset

from Training.LitModule import LitModule

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'=== DEVICE : {device} ===')

    # Setting logger
    wandb_logger = WandbLogger(project=projec_name, log_model=True)

    # Load documentation dataset
    doc_dataset = DocumentDataset(size=(5500, 5500), blur_kernel=(4,4))
    doc_dataset.load_dataset()

    # I take train dataset and validation dataset
    len_dataset = doc_dataset.__len__()
    train_ratio = 0.8
    val_ratio = 0.2

    num_train = int(len_dataset * train_ratio)
    num_val = len_dataset - num_train
    print(f'Samples number in to dataset : {len_dataset}')
    print(f'Samples in to train set : {num_train}')
    print(f'Samples in to validation set : {num_val}')

    train_set, val_set = doc_dataset.split_dataset(num_train, num_val)

    num_classes = 3
    lr = 1e-4
    weights_decay = 1e-5
    train_epochs = 100

    model = LitModule(num_classes=num_classes, lr=lr, weights_decay=weights_decay)
    trainer = pl.Trainer(
        max_epochs=train_epochs,
        logger=wandb_logger,
        accelerator='auto',
        devices=device,
        log_every_n_steps=10,
        deterministic=True,
        enable_checkpointing=True
    )





    
    

