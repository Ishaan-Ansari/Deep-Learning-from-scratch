"""
In this section we'll implement
Batch, Stochastic & Mini-Batch GD
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# set random value for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SimpleNet(nn.Module):
    def __init__(self, input_size=2, hidden_size=10, output_size=1):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def generate_sample_data(n_samples=1000):
    """Generate some sample data for regression"""
    X = torch.randn(n_samples, 2)

    y = (X[:, 0]**2 + X[:, 1]**2 + 0.5*X[:, 0]*X[:, 1]).unsqueeze(1)
    y += 0.1 * torch.randn(n_samples, 1)

    return X, y

def compute_loss_and_gradients(model, X, y, loss_fn):
    """Compute loss and gradient for given data"""
    model.zero_grad()
    predictions = model(X)
    loss = loss_fn(predictions, y)
    loss.backward()
    return loss.item()

def get_gradients(model):
    """Extract gradients from model parameters"""
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.clone())
    return gradients

def apply_gradients(model, gradients, learning_rate):
    """Apply gradients to model parameters"""
    for param, grad in zip(model.parameters(), gradients):
        param.data -= learning_rate * grad

