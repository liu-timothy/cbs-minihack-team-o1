import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 100, 64)  # Adjust the input features of fc1 based on your sequence length
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.one_hot(x, num_classes=4).float()  # Assuming one-hot encoding for DNA sequences
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 32 * 100)  # Flatten the tensor before passing to the linear layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
