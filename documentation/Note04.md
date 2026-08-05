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
