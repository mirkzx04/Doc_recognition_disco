import os
import cv2
import numpy as np

from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision import models, transforms


dataset_path = r"\\10.5.1.36\dataset_IA\dataset_pdf_Solo_Rinnovi\images"

from Dataset_classes.DocDataset import DocumentDataset

def freeze_model(model):
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False

if __name__ == "__main__":
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

    # Load pre train model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_classes = 3

    # Freeze model except for linear layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    freeze_model(model)

    # Set hyperparameters for optimizer
    lr = 0.0001
    weights_decay = 1e-5

    train_epochs = 100

    # Set optimizer, loss and lr_scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.fc.parameters(), lr=lr, weight_decay=weights_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=train_epochs)



    
    

