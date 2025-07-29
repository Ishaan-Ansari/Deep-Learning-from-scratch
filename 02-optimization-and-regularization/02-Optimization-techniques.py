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

class SimpleNeuralNet(nn.Module):
    def __init__(self, input_size=2, hidden_size=10, output_size=1):
        super(SimpleNeuralNet, self).__init__()
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
        """Single training step using one sample at a time"""
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

class MiniBatchGradientDescent:
    """
       Mini-batch gradient descent - Uses some samples at a time
       (usually a batch size of 32 is taken)

       Pros:
       - Good balance between speed and stability
       - Efficient use of vectorization
       - Less memory usage than batch, more stable than SGD

       Cons:
       - Need to tune batch size
       - Still some noise in gradients
       """

    def __init__(self, model, learning_rate=0.01, batch_size=32):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_history = []

    def train_step(self, x_batch, y_batch, loss_fn):
        """Single training step using mini-batch"""
        loss = compute_loss_and_gradients(self.model, x_batch, y_batch, loss_fn)

        # update parameters
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    param.data -= self.learning_rate * param.grad

        return loss


    def train(self, X_train, y_train, epochs=100):
        loss_fn = nn.MSELoss()

        # Create Dataloader for mini-batching
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        print("Starting Mini-Batch Gradient Descent...")
        for epoch in range(epochs):
            epoch_losses = []

            # Process mini-batches
            for x_batch, y_batch in dataloader:
                loss = self.train_step(x_batch, y_batch, loss_fn)
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses)
            self.loss_history.append(avg_loss)

            if epoch % 20 == 0:
                print(f"Epoch {epoch} Loss: {avg_loss: .4f}")

        return self.loss_history


def compare_algorithms():
    """Compare all three gradient descent variants"""

    # Generate data
    X, y = generate_sample_data(n_samples=1000)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Create three identical models
    model_batch = SimpleNeuralNet()
    model_sgd = SimpleNeuralNet()
    model_mini = SimpleNeuralNet()

    # Copy weights to ensure fair comparison
    model_sgd.load_state_dict(model_batch.state_dict())
    model_mini.load_state_dict(model_batch.state_dict())

    epochs = 50

    # Train with Batch GD
    batch_trainer = BatchGradientDescent(model_batch, learning_rate=0.01)
    batch_losses = batch_trainer.train(X, y, epochs=epochs)

    # Train with SGD
    sgd_trainer = StochasticGradientDescent(model_sgd, learning_rate=0.01)
    sgd_losses = sgd_trainer.train(X, y, epochs=epochs)

    # Train with Mini-batch GD
    mini_trainer = MiniBatchGradientDescent(model_mini, learning_rate=0.01, batch_size=32)
    mini_losses = mini_trainer.train(X, y, epochs=epochs)

    # Plot comparison
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(batch_losses, label='Batch GD', linewidth=2)
    plt.title('Batch Gradient Descent')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(sgd_losses, label='Stochastic GD', linewidth=2, alpha=0.7)
    plt.title('Stochastic Gradient Descent')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(mini_losses, label='Mini-batch GD', linewidth=2)
    plt.title('Mini-batch Gradient Descent')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(batch_losses, label='Batch GD', linewidth=2)
    plt.plot(sgd_losses, label='Stochastic GD', alpha=0.7)
    plt.plot(mini_losses, label='Mini-batch GD', linewidth=2)
    plt.title('Comparison of All Methods')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print final losses
    print(f"\nFinal Losses:")
    print(f"Batch GD: {batch_losses[-1]:.4f}")
    print(f"Stochastic GD: {sgd_losses[-1]:.4f}")
    print(f"Mini-batch GD: {mini_losses[-1]:.4f}")


def demonstrate_key_concepts():
    """Demonstrate key concepts with simple examples"""

    print("=== KEY CONCEPTS DEMONSTRATION ===\n")

    # Create sample data
    X, y = generate_sample_data(n_samples=100)

    print("1. BATCH SIZE IMPACT:")
    print(f"   Full dataset: {X.shape[0]} samples")
    print(f"   Batch GD: Uses all {X.shape[0]} samples per update")
    print(f"   Stochastic GD: Uses 1 sample per update")
    print(f"   Mini-batch GD: Uses 32 samples per update")

    print("\n2. UPDATE FREQUENCY:")
    print(f"   Batch GD: 1 update per epoch")
    print(f"   Stochastic GD: {X.shape[0]} updates per epoch")
    print(f"   Mini-batch GD: {X.shape[0] // 32} updates per epoch")

    print("\n3. MEMORY USAGE:")
    print(f"   Batch GD: Loads entire dataset ({X.shape[0]} samples)")
    print(f"   Stochastic GD: Loads 1 sample at a time")
    print(f"   Mini-batch GD: Loads 32 samples at a time")

    print("\n4. CONVERGENCE CHARACTERISTICS:")
    print(f"   Batch GD: Smooth, stable convergence")
    print(f"   Stochastic GD: Noisy, fast but unstable")
    print(f"   Mini-batch GD: Balanced - reasonably smooth and fast")


# Run the demonstration
if __name__ == "__main__":
    demonstrate_key_concepts()
    print("\n" + "=" * 50)
    compare_algorithms()