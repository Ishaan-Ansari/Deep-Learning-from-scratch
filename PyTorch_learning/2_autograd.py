"""
PyTorch automatic differentiation engine (autograd)
"""

import torch
import torch.nn.functional as F

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)    # enable gradient tracking
b = torch.tensor([0.5], requires_grad=True) 

z = x1 * w1 + b
a = torch.sigmoid(z)

loss = F.binary_cross_entropy(a, y)
print(loss)