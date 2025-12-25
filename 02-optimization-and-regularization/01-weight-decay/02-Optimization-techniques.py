"""
This file contains optimization techniques implementations for neural networks.
1. Batch Gradient Descent
2. Stochastic Gradient Descent
3. Mini-Batch Gradient Descent
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset  

class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(input_size, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.

        Parameters:
            x (torch.Tensor): Input tensor to the neural network.

        Returns:
            torch.Tensor: Output logits of the neural network.
        """
        logits = self.layers(x)
        return logits
    

class BatchGradientDescent:
    """
    Batch Gradient Descent: Uses the ENTIRE dataset to compute gradients and update weights.
    """
    def __init__(self, model: NeuralNetwork, learning_rate: float = 0.01):
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    def train_epoch(self, X_train: torch.Tensor, y_train: torch.Tensor):
        """
        Train for ONE epoch using ENTIRE dataset at once.

        Flow:
        1. Zero out previous gradients.
        2. Forward pass on all training data.
        3. Compute loss.
        4. Backward pass to compute gradients.
        5. Update weights.

        Args:
            X_train (torch.Tensor): Training input data.
            y_train (torch.Tensor): Training target labels.

        Returns:
            float: Loss value for the epoch.
        """
        self.model.train()

        # 1. Zero out previous gradients
        self.optimizer.zero_grad()

        # 2. Forward pass on all training data
        logits = self.model(X_train)

        # 3. Compute loss
        loss = F.cross_entropy(logits, y_train)

        # 4. Backward pass to compute gradients
        loss.backward()

        # 5. Update weights
        self.optimizer.step()

        return loss.item()
    
    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int = 100):
        """
        Train the model for a specified number of epochs.

        Args:
            X_train (torch.Tensor): Training input data.
            y_train (torch.Tensor): Training target labels.
            epochs (int): Number of epochs to train.
        """
        for epoch in range(epochs):
            loss = self.train_epoch(X_train, y_train)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

class StochasticGradientDescent:
    """
    Stochastic Gradient Descent: Uses ONE data point at a time to compute gradients and update weights.
    """
    def __init__(self, model: NeuralNetwork, learning_rate: float = 0.01):
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    def train_epoch(self, X_train: torch.Tensor, y_train: torch.Tensor):
        """
        Train for ONE epoch using ONE data point at a time.

        Flow (repeated N times for N data points):
        1. Pick ONE random sample.
        2. Zero out previous gradients.
        3. Forward pass on the sample.
        4. Compute loss.
        5. Backward pass to compute gradients.
        6. Update weights.

        Args:
            X_train (torch.Tensor): Training input data.
            y_train (torch.Tensor): Training target labels.
        Returns:
            float: Average loss value for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_samples = X_train.size(0)

        # Create random permutation of indices for shuffling
        indices = torch.randperm(n_samples)

        for i in indices:
            # 1. Pick ONE random sample
            x_single = X_train[i:i+1]
            y_single = y_train[i:i+1]

            # 2. Zero out previous gradients
            self.optimizer.zero_grad()

            # 3. Forward pass on the sample
            logits = self.model(x_single)

            # 4. Compute loss
            loss = F.cross_entropy(logits, y_single)

            # 5. Backward pass to compute gradients
            loss.backward()

            # 6. Update weights
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / n_samples
    
    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int = 100):
        """
        Train the model for a specified number of epochs.

        Args:
            X_train (torch.Tensor): Training input data.
            y_train (torch.Tensor): Training target labels.
            epochs (int): Number of epochs to train.
        """
        for epoch in range(epochs):
            loss = self.train_epoch(X_train, y_train)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

class MiniBatchGradientDescent:
    """
    Mini-Batch Gradient Descent: Uses a SMALL BATCH of data points to compute gradients and update weights.
    """
    def __init__(self, model: NeuralNetwork, learning_rate: float = 0.01, batch_size: int = 32):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    def train_epoch(self, train_loader: DataLoader):
        """
        Train for ONE epoch using MINI-BATCHES of data.

        Flow (repeated for each mini-batch):
        1. Get a mini-batch of samples.
        2. Zero out previous gradients.
        3. Forward pass on the mini -batch.
        4. Compute loss.
        5. Backward pass to compute gradients.
        6. Update weights.


        Args:
            X_train (torch.Tensor): Training input data.
            y_train (torch.Tensor): Training target labels.

        Returns:
            float: Average loss value for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for X_batch, y_batch in train_loader:
            # 1. Zero out previous gradients
            self.optimizer.zero_grad()

            # 2. Forward pass on the mini-batch
            logits = self.model(X_batch)

            # 3. Compute loss
            loss = F.cross_entropy(logits, y_batch)

            # 4. Backward pass to compute gradients
            loss.backward()

            # 5. Update weights
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / n_samples
    

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int = 100):
        """
        Train the model for a specified number of epochs.

        Args:
            X_train (torch.Tensor): Training input data.
            y_train (torch.Tensor): Training target labels.
            epochs (int): Number of epochs to train.
        """

        dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(epochs):
            loss = self.train_epoch(train_loader)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")


        