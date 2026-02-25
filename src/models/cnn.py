import torch.nn.functional as F
from torch import nn


class EdgeCNNModel(nn.Module):
    def __init__(self, n_features=9):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 56, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
