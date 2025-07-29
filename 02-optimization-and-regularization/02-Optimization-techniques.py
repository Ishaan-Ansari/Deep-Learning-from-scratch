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
    model.zero_grad()   # clears any accumulation on params in a prev. pass
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


class BatchGradientDescent:
    """
    Batch gradient descent - Uses ALL training data for each update

    Pros:
    - Stable convergence
    - Exact gradient computation

    Cons:
    - High latency
    - Requires entire dataset memory

    """

    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate
        self.loss_history = []

    def train_step(self, X_train, y_train, loss_fn):
        """Single training step using entire dataset"""
        loss = compute_loss_and_gradients(self.model, X_train, y_train, loss_fn)


        # update parameters using computed gradients
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    param.data -= self.learning_rate * param.grad

        self.loss_history.append(loss)
        return loss

    def train(self, X_train, y_train, epochs=100):
        loss_fn = nn.MSELoss()

        print("Starting Batch Gradient Descent...")
        for epoch in range(epochs):
            loss = self.train_step(X_train, y_train, loss_fn)

            if epoch % 20 == 0:
                print(f"Epoch {epoch} Loss: {loss: .4f}")

        return self.loss_history


class StochasticGradientDescent:
    """
    Stochastic gradient descent - Uses one sample at a time

    Pros:
    - Very fast updates
    - Can escape local minima due to noise
    - Memory efficient

    Cons:
    - Noisy convergence
    - May not converge to exact minima
    """

    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate
        self.loss_history = []

    def train_step(self, x_sample, y_sample, loss_fn):
        """Single traing step using one sample at a time"""
        if x_sample.dim() == 1:
            x_sample = x_sample.unsqueeze(0)
        if y_sample.dim() == 1:
            y_sample = y_sample.unsqueeze(0)

        loss = compute_loss_and_gradients(self.model, x_sample, y_sample, loss_fn)

        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    param.data -= self.learning_rate * param.grad

        return loss

    def train(self, X_train, y_train, epochs=100):
        loss_fn = nn.MSELoss()

        print("Starting Stochastic Gradient Descent...")
        for epoch in range(epochs):
            epoch_losses = []

            # shuffle data in each epoch
            indices = torch.randperm(X_train.size(0))

            # process one sample at time
            for i in indices:
                x_sample = X_train[i]
                y_sample = y_train[i]

                loss = self.train_step(x_sample, y_sample, loss_fn)
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses)
            self.loss_history.append(avg_loss)

            if epoch % 20 == 0:
                print(f"Epoch {epoch} Loss: {avg_loss: .4f}")

        return self.loss_history
