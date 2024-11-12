import torch
from torch import nn
from torch.nn import functional as F
"""
My baseline was a CNN project I did for a FreeCodeCamp certification: 
https://github.com/emanueleiacca/FreeCode-Camp-Machine-Learning-with-Python/blob/main/fcc_cat_dog.ipynb
Since the dataset was having problems in terms on quality and quantity of the data
I focused a lot on the best Data Augmentation tecniques and the best performing layer structure.
This will be our starting point that we will adjust according to the needs of CIFAR-10 dataset
"""
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        # Convolutional layers 
        """
        Fixed parameters are:
        Input channel = 3 cause the imgs have RGB colors
        Standard Kernel 3x3
        Padding = 1 to preserve img size (between input e output)
        """
        # First Layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) # We use 32 filters in the first layer to capture low-level patterns like edges and textures
        self.bn1 = nn.BatchNorm2d(32) #Speeds up convergence by reducing internal covariate shift, it also regularize the model, reducing the risk of overfitting

        # Second Layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Increasing to 64 filters captures more complex features, like shapes and object parts
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third Layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # We increase to 128 filters as deeper layers need to capture high-level patterns, like objects
        self.bn3 = nn.BatchNorm2d(128)

        # Fourth (Additional) Layer
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # We try to capture even more complex features
        self.bn4 = nn.BatchNorm2d(256)

        
        # Dropout layer
        # dropout parameters changed from 0.4 to 0.3
        self.dropout = nn.Dropout(0.3) # randomly "drops" 30% of the neurons during training, forcing the network to generalize better and avoid overfitting
        
        # Fully connected layers
            # #The output of the last convolutional layer is reshaped into a 1D vector to feed into the fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256) # Reduces the feature vector to 256 units, original was (number of filters * reduced spatial dimensions from pooling[0] * reduced spatial dimensions from pooling[1])
        self.fc2 = nn.Linear(256, 10) # output vector of length 10 (classes)

    def forward(self, x):
        """
        Each convolutional block contains:
        convolutional layer,
        batch normalization, 
        ReLU activation, 
        max pooling (2 is stride value, it control the number of jumps a feature map must make per max pool operation)
        """
        # First Convolutional Block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        # Second Convolutional Block
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # Third Convolutional Block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        # Fourth (Additional) Convolutional Block
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)

        # Dropout
        x = self.dropout(x)  # applying dropout at different stages allows to avoids learning too specific patterns from the training data
        
        # Flatten for Fully Connected Layers
        #x = x.view(-1, 128 * 4 * 4) # Instead of flattening the feature maps to a vector
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1) # Global Average Pooling reduce overfitting (so more generalizzation)
        
        # Fully Connected Layers
        # First FC layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x) #
        # Second FC layer
        x = self.fc2(x) 
        
        return x #outputs the raw class scores for the 10 classes
