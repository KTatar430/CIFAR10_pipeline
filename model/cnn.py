# Definition of a convolutional neural network for image classification
# The network must have 3 input neurons, corresponding to the 3 color channels (RGB) of the input images
# The network must have 10 output neurons, corresponding to the 10 classes in the CIFAR-10 dataset

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(24, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # conv1 -> relu -> pool -> conv2 -> relu -> pool -> conv3 -> relu -> flatten -> fc1 -> relu -> fc2
        x = self.pool(F.relu(self.conv1(x)))   # (3, 32, 32) -> (16, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))   # (16, 16, 16) -> (24, 8, 8)
        x = F.relu(self.conv3(x))              # (24, 8, 8) -> (32, 8, 8)
        x = torch.flatten(x, 1)                # (batch, 2048)
        x = F.relu(self.fc1(x))                # (batch, 2048) -> (batch, 256)
        x = self.fc2(x)                        # (batch, 256) -> (batch, 10)

        return x
