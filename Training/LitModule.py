import pytorch_lightning as pl

from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision import models, transforms

class LitModule(pl.LightningModule):
    def __init__(self, num_classes, lr = 1e-4, weights_decay = 1e-5):
        super().__init__()

        self.save_hyperparameters()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.criterion = nn.CrossEntropyLoss()

        self.lr = lr
        self.weights_decay = weights_decay

    def forward(self, X):
        return self.model(X)
    
    def training_step(self, batch, batch_idx):
        data, labels = batch
        logits = self(data)
        loss = self.criterion(logits, labels)

        acc = (logits.argmas(dim=1) == labels).float().mean()

        self.log('train_loss', loss, on_step = True, on_epoch = True, prog_bar = True)
        self.log('train_acc', acc, on_step=True, on_epoch= True, prog_bar= True )

        return loss
    
    def validation_step(self, batch, batch_idx):
        data, labels = batch
        logits = self(data)
        loss = self.criterion(logits, labels)

        acc = (logits.argmas(dim=1) == labels).float().mean()

        self.log('val_loss', loss, on_step = True, on_epoch = True, prog_bar = True)
        self.log('val_acc', acc, on_step=True, on_epoch= True, prog_bar= True )

        
        

