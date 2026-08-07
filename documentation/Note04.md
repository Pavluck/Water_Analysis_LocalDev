# Building the Model

Models from torchvision are pre-trained on ImageNet, which is a large dataset of images with 1000 classes. 
The pre-trained weights allow the model to leverage learned features from a wide variety of images, which can be useful for transfer learning. Start with a template, and build upward.
Working in small steps helps with debugging, and feature engineering. The steps can be lumped, but in my case, I built the model with a base template model, writing small chunks of code and reading the upsides and downsides of CNN and ResNet.

From reading, CNNs can have issues with vanishing gradients and overfitting. 
I researched the advantages of implementing residual connections. 
One advantage is that ResNet focuses on global context, which is good since if water is in a glass, it can use the context of the background to determine if the water is clear or murky. 
It will also make the model less susceptible to being tricked by solid blue items being seen as potable water within its classification. 
Of course, this means that the ResNet does not focus well on isolated spectrum's within an image or video, which is a tradeoff from the ResNet and CNN models.

Below is an example comparing the performance of a CNN model with a ResNet model. The graph below tests the models using an animal classification dataset.
<img width="909" height="487" alt="image" src="https://github.com/user-attachments/assets/6c31bfd5-0db0-4680-b97c-03ab79482775" />
[Reference: https://medium.com/@leonardofonseca.r/a-practical-comparison-between-cnn-and-resnet-architectures-a-focus-on-attention-mechanisms-cee7ec8eca55]

## Starting the Model
In this case, we are using ResNet18 as the backbone for our water potability classification model. 
The final fully connected layer is replaced with a custom head for binary classification (potable vs. non-potable). 
The model is initialized with pre-trained weights from ImageNet, and the final layer is modified to output the desired number of classes, 2 for binary classification

Note that the parameter, nn.Module, is a base class for all neural network modules in PyTorch. 
It provides a way to define and manage layers, parameters, and forward passes in a structured manner. 
By subclassing nn.Module, we can create custom models with specific architectures and behaviors.

```
class WaterCNN(nn.Module):
    """
    Takes in a pretrained ImageNet model from torchvision (as a template) and prepares the data for training and testing
    """

    def __init__(self, num_classes=2):
        """
        2 class, which represent the different decisions (potable water and non-potable water)
        """
        # Call the constructor of the parent class nn.Module to initialize the model and its parameters.
        super().__init__()

    def forward(self, tensor):
        """
        Pass input tensor through the ResNet backbone which returns a prediction.
        """
        return self.backbone(tensor)
```

## Model Architecture 
The class initializes the model with a ResNet18 backbone and a classification head for binary classification with that init function.
The final layer is fully connected, which gets replaced with a new head consisting of a linear layer, ReLU activation, dropout for regularization, and another linear layer to output the desired number of classes.

We can set up the backbone and prepares the model for training to identify characteristics of potable water.
This can be done by extracting the number of input features for the final fully connected layer of the ResNet18 backbone.

Then the defined new classification head will replace the original template, and be a fully connected layer

```
import torch.nn as nn

class WaterCNN(nn.Module):
    """
    Takes in a pretrained ImageNet model from torchvision (as a template) and prepares the data for training and testing
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        num_features = self.backbone.fc.in_features
```

## Building a better backbone
nn.Linear can be called twice in the classification head: first to reduce the feature dimension from num_features to 256, and then to map the 256 features to the desired number of output classes (num_classes). The first nn.Linear layer learns a transformation of the features extracted by the backbone, while the second nn.Linear layer produces the final class scores for classification.

```
import torch.nn as nn

class WaterCNN(nn.Module):
    """
    Set up the base model to prepare it for training and eventually, testing.
    Receives the pretrained ImageNet model from torchvision to copy into a unique instance
    """
    def __init__(self, num_classes=2):
        """
        Initializes the WaterCNN model with a ResNet18 backbone.
        """
        super().__init__() # Call the constructor of the parent class nn.Module for initialization
        self.backbone = models.resnet18(pretrained=True)
        num_features = self.backbone.fc.in_features # input features for the ResN backbone

```


## References:
[1] [Reference: https://medium.com/@leonardofonseca.r/a-practical-comparison-between-cnn-and-resnet-architectures-a-focus-on-attention-mechanisms-cee7ec8eca55]
