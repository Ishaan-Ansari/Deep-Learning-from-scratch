"""
Creating efficient data loaders in PyTorch
"""
import torch

X_train = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
    [7.0, 8.0],
    [9.0, 10.0]
])

y_train = torch.tensor([0, 1, 0, 1, 0])

X_test = torch.tensor([
    [11.0, 12.0],
    [13.0, 14.0]
])

y_test = torch.tensor([1, 0])

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y
    
    def __len__(self):
        return len(self.features)
    
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

"""
NOTE: In the __init__ method, we set up attributes that we can access later in the __getitem__ and __len__ methods. 
This could be file paths, file objects, database connectors, and so on. 
Since we created a tensor dataset that sits in memory, we are simply assigning X and y to these attributes, which are placeholders for our tensor objects.
"""

from torch.utils.data import DataLoader

torch.manual_seed(42)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

for idx, (x, y) in enumerate(train_dataloader):
    print(f"Batch {idx+1}")
    print(f"Features:\n{x}")
    print(f"Labels:\n{y}\n")