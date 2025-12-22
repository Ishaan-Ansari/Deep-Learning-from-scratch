"""
Neural networks in PyTorch
"""
import torch
import torch.nn.functional as F

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs),
        )
        
    def forward(self, x):
        logits = self.layers(x)
        return logits
    
# create a neural network instance
model = NeuralNetwork(num_inputs=10, num_outputs=1)
# print the model architecture
print(model)

# Next, let’s check the total number of trainable parameters of this model:
total_params = sum(
    param.numel() for param in model.parameters() if param.requires_grad
)
print(f'Total trainable parameters: {total_params}')