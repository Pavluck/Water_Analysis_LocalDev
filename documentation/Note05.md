# Training

For the training setup for CNN model for water potability classification using a ResNet18 will include global parameters to be used for the classes, such as the number of epochs, the weight decay, learning rate, etc.

## Set Up
Global parameters will be established for the training.
An epoch is defined as a single cycle where every sample in the training dataset is processed by the model (for both a forward and backward pass). Too few or too many epochs can lead to a decay in accuracy. This will be set to 20 for now, a safe standard.

The Learning rate is normally a decimal to represent a percentage. The standard initial learning rate for training a ResNet model from scratch on large datasets is 0.1. Since our Resnet comes pretrained from the Pytorch template, the learning rate will be a fraction of that to avoid destroying learned features. 

```
# ~~ Training Setup ~~
EPOCHS = 20
LEARNING_RATE = 0.001
```

## Training Parameters
Weight decay for regularization is added to the optimizer and set to 1e-4 to help prevent overfitting and improve generalization. 1e-4 is a common default for many models.

In addition, gradient clipping can be applied. It is a technique to limit the maximum value of gradients during backpropagation, which helps stabilize training. To prevent exploding gradients during training, a value of 1.0 will be set.

```
# ~~ Training Setup ~~
...
WEIGHT_DECAY = 1e-4
NORMALIZATION = 1.0 
```

### Backbone Optimization
We can leverage the learned features from the pre-trained model while focusing on training the new classification head by a technique called freezing the backbone. After the specified number of epochs, the backbone is unfrozen to allow fine-tuning of the entire model for better performance on the new task.

By setting the number of epochs to freeze the backbone during training allows the model to learn the new classification head before fine-tuning the backbone. After this many epochs, the backbone will be unfrozen and trained along with the head. backbone freezing is a common technique in transfer learning, where a pre-trained model is used as a starting point for a new task.

```
# ~~ Training Setup ~~
NAME = "CNNv2.5.pth"
WEIGHT_DECAY = 1e-4
NORMALIZATION = 1.0
BACKBONE_FREEZE = 5    # <~ freeze every 5 epochs
```

The backbone is less sensitive to learning rate changes than the head, so we apply a lower learning rate to the backbone during training. This factor is multiplied by the base learning rate to set the learning rate for the backbone parameters.

```
BACKBONE_LR = 0.1
```

# Optimization

The torch.device function utilizes the NVIDIA driver to speed up training. Includes an if statement for those who desire to run the code with machines that do not have the NVIDIA GPU.

Since cuda runs well on binary exponents, 32 samples per weight update optimizes the GPU without too much memory consumption.

The pytorch import handles memory allocation and parallel processing via thread blocks on NVIDIA's CUDA architecture. A 32 block size can also be known as a wrap, and behaves well with the gpu's available cores without causing problems for the code being run on devices that cannot use the gpu. 
