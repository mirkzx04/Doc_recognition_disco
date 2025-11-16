<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
# DiscoLazio Document Classification 

## Dataset 
Our dataset is composed of the student document images. The images has differents size. The preprocessing of the dataset consisted of the resizing and standardization of the images.
To do this i had implemented a DocDataset class, it take all image from dataset, resize and standardize it.

Classes in the img_prepoc folder are for image preprocessing. But we aren't using it because in our tests with preprocessed images, we didn't see significant improvement.
So as preprocessing we are only using resizing and standardization of the images. The resizing is necessary because the network takes 224x224 images as input.
# DiscoLazio Document classification 

## Model

As a model we are using a pretrained ResNet. So we are fine-tuning it to adapt it to our dataset
To fine-tune the ResNet we froze all the networks weights except for the linear layer weights. We then unfroze the layer 4 and layer 3 parameters at different epochs.

We used AdamW as optimizer, LambdaLR as warm-up scheduler and CosineAnnealingLR.

## Training

We got a high accuracy in only 50 epochs, then I delivered the model to DiscoLazio and they have load the model on their server
