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
        x = F.one_hot(x, num_classes=4).permute(0, 2, 1).float()  # One-hot encoding and permuting to fit Conv1d
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x