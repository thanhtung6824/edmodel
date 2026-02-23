import torch.nn.functional as F
from torch import nn


class EdgeCNNModel(nn.Module):
    def __init__(self, n_features=9, window=60):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)

        # Sau 2 conv layers: 60 - 2 - 2 = 56 timesteps
        self.fc = nn.Linear(64 * 56, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, 9, 60)
        x = F.relu(self.conv1(x))  # (batch, 32, 58)
        x = F.relu(self.conv2(x))  # (batch, 64, 56)
        x = x.flatten(1)  # (batch, 64*56) = (batch, 3584)
        x = self.fc(x)  # (batch, 1)
        return x
