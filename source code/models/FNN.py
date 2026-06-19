"""
FNN.py

Contains Feed-Forward Neural Networks (FNN) for regression tasks:
- BaselineFNN: simple MLP with 2 hidden layers 

Loss used for training: Mean Squared Error (MSE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineFNN(nn.Module):
    """
    Baseline Feed-Forward Neural Network for regression.
    - input_size: number of input features
    - hidden_sizes: list with sizes of hidden layers
    - output_size: regression output size (usually = 1)
    """
    def __init__(self, input_size, hidden_sizes=[128, 64], output_size=1):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)  #regression output

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  #last layer = linear (for regression)
        return x