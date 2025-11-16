import os
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from Dataset_classes.DocDataset import DocumentDataset
from Training.LitModule import LitModule

from collections import Counter

def error_analysis(error_matrix, num_classes):
    if error_matrix.shape[0] > 0: # Check if there are any errors
        if len(error_matrix.shape) == 1 or isinstance(error_matrix[0], list):
            # Convert matrix in [epochs, error_class]
            processed_matrix = []
            for epoch_errors in error_matrix:
                if isinstance(epoch_errors, list):
                    error_counts = Counter(epoch_errors)
                    epoch_row = [error_counts.get(class_idx, 0) for class_idx in range(num_classes)]
                    processed_matrix.append(epoch_row)
                else:
                    processed_matrix.append(epoch_errors)
            error_matrix = np.array(processed_matrix)

def plot(error_matrix, num_classes):
    # Create plot
    plt.figure(figsize=(10, 6))

    # Define color per classes
    colors = plt.cm.table10(np.linspace(0, 1, num_classes))
    if num_classes > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
    
    epochs = range(1, error_matrix.shape[0] + 1)

    # Plot each class with its own color
    for class_idx in range(num_classes):
        errors_per_epoch = error_matrix[:, class_idx]
        class_name = name_classes[class_idx] if class_idx < len(name_classes) else f'Classe_{class_idx}'

        plt.plot(
            epochs,
            errors_per_epoch,
            color = colors[class_idx],
            marker='o',
            marksize = 4,
            linewidth=2,
            label=f'{class_name}'
        )

    plt.xlabel('Epochs', fontsize = 15)
    plt.ylabel('Number of Errors', fontsize = 15)
    plt.title('Errors per Class over Epochs', fontsize = 18)
    plt.grid(True, alpha = 0.3)

    plt.xticks(range(1, len(epochs) + 1, max(1, len(epochs)//10)))

    # Save plot
    plt.tight_layout()
    plot_filename = 'error_analysis_plot.png'
    plt.savefig(plot_filename)

    plt.show()

if __name__ == "__main__":
    prject_name = 'Disco_Doc'

    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    print(f'=== DEVICE : {device} ===')

    # Setting logger
    wandb_logger = WandbLogger(project=prject_name, log_model=True, name = 'Disco_Doc_v2')

    # Load documentation dataset
    doc_dataset = DocumentDataset(size=(224, 224), blur_kernel=(4,4), dataset_path=dataset_path)
    doc_dataset.create_dataset()

    # DEBUG: Controlla le labels PRIMA del mapping
    name_classes = np.unique(doc_dataset.labels)
    num_classes = len(name_classes)
    print(f"Classi uniche ORIGINALI: {name_classes}")
    print(f"Numero classi: {num_classes}")
    print(f"Min label ORIGINALE: {np.min(doc_dataset.labels)}")
    print(f"Max label ORIGINALE: {np.max(doc_dataset.labels)}")
    
    # CORREZIONE: Rimappa le labels a [0, n_classes-1] PRIMA del split
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(name_classes)}
    print(f"Mapping labels: {label_mapping}")
    
    # Applica il mapping al dataset
    doc_dataset.labels = [label_mapping[label] for label in doc_dataset.labels]
    
    # Verifica dopo il mapping
    print(f"Dopo mapping - Min: {np.min(doc_dataset.labels)}, Max: {np.max(doc_dataset.labels)}")
    
    # Ora fai il split DOPO il remapping
    train_ratio, val_ratio = 0.8, 0.2
    train_set, val_set = doc_dataset.split_dataset(train_ratio, val_ratio)
     
    # Ricrea i DataLoader con batch_size corretto
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)

    lr = 1e-4
    weights_decay = 1e-4
    train_epochs = 100

    checkpoint_callback = ModelCheckpoint(
        save_top_k = 1,
        save_last = False,
        mode='min',
        monitor='avg_val_loss',
        filename='best_checkpoint'
    )
    early_stopping_callback = EarlyStopping(
        monitor = 'avg_val_loss',
        min_delta = 0.001,
        patience = 10,
        verbose = True,
        mode = 'min',
        check_finite = True,
    )

    model = LitModule(device=device,name_classes=name_classes, num_classes=num_classes, lr=lr, weights_decay=weights_decay)
    trainer = pl.Trainer(
        max_epochs=train_epochs,
        logger=wandb_logger,
        accelerator=device,
        devices=1 if torch.cuda.is_available() else 'auto',
        enable_checkpointing=True,
        gradient_clip_algorithm='norm',
        gradient_clip_val=1.0,
        num_sanity_val_steps=0,
        callbacks = [checkpoint_callback, early_stopping_callback],
        precision = '16-mixed'
    )
    trainer.fit(model, train_loader, val_loader)

    error_matrix = trainer.get_error_matrix()
    error_matrix = error_analysis(error_matrix, num_classes)










