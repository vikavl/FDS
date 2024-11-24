import torch.nn as nn
import torch

class PoorPerformingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        ##############################
        ###     CHANGE THIS CODE   ###
        ##############################  
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(8 * 8 * 8, 28)

    def forward(self, x):
        # 32x32 color image: 3 channels RGB
        # input: 32x32x3
        # after conv1: 32x32x4
        # after max pooling: 16x16x4 
        x = self.pool(self.relu1(self.conv1(x)))
        # input: 16x16x4 
        # after conv2: 16x16x8
        # after max pooling: 8x8x8
        x = self.pool(self.relu2(self.conv2(x)))
        x = x.view(-1, 8 * 8 * 8)
        x = self.fc1(x)
        return x
