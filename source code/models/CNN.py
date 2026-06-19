
"""
CNN.py

Contains CNN architectures for GTSRB traffic sign classification:
- BaselineCNN: standard CNN with no dropout
- DropoutCNN: CNN with configurable dropout for regularization
"""


import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    """Baseline CNN with 2 convolutional layers and 2 fully connected layers.
       No dropout or regularization applied.
       Designed for the GTSRB traffic sign classification dataset (43 classes).
    """
    def __init__(self):
        super().__init__()
        #Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 max pooling

        #Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 256)  # Flattened feature size after pooling
        self.fc2 = nn.Linear(256, 43)  # 43 traffic sign classes

    def forward(self, x):
        #Forward pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        #Flatten feature map for fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  #Output logits for 43 classes
        return x


class DropoutCNN(nn.Module):
    """CNN with dropout for regularization to reduce overfitting.
       Dropout probability can be set via `dropout_p`.
    """
    def __init__(self, dropout_p=0.5):
        super().__init__()
        #Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        #Fully connected layers with dropout
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(256, 43)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
