# Training

For the training setup for CNN model for water potability classification using a ResNet18 will include global parameters to be used for the classes, such as the number of epochs, the weight decay, learning rate, etc.

## Set Up
Global parameters will be established for the training.
An epoch is defined as a single cycle where every sample in the training dataset is processed by the model (for both a forward and backward pass). Too few or too many epochs can lead to a decay in accuracy. This will be set to 20 for now, a safe standard.

The Learning rate is normally a decimal to represent a percentage. The standard initial learning rate for training a ResNet model from scratch on large datasets is 0.1. Since our Resnet comes pretrained from the Pytorch template, the learning rate will be a fraction of that to avoid destroying learned features. 
