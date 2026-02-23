import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class BTCDataset(Dataset):
    def __init__(self, data, window=60):
        self.data = data
        self.window = window

    def __len__(self):
        return len(self.data) - self.window

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window]
        y = self.data[idx + self.window, 3]
        return torch.FloatTensor(x), torch.FloatTensor([y])
