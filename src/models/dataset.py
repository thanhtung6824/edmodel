import torch
from torch.utils.data import Dataset

class BTCDataset(Dataset):
    def __init__(self, scaled_data, raw_data, window=60, lookahead=6, threshold=0.005):
        self.data = scaled_data
        self.raw_data = raw_data
        self.window = window
        self.lookahead = lookahead    # 6 candles 4h = 24h
        self.threshold = threshold

    def __len__(self):
        return len(self.data) - self.window - self.lookahead

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window]

        current_close = self.raw_data[idx + self.window - 1, 3]
        future_closes = self.raw_data[idx + self.window : idx + self.window + self.lookahead, 3]

        future_max = future_closes.max()
        future_min = future_closes.min()

        up_potential   = (future_max - current_close) / (current_close + 1e-8)
        down_potential = (current_close - future_min) / (current_close + 1e-8)

        # Hướng nào có potential lớn hơn?
        if up_potential > down_potential and up_potential > self.threshold:
            direction = 2   # UP
        elif down_potential > up_potential and down_potential > self.threshold:
            direction = 0   # DOWN
        else:
            direction = 1   # SIDEWAY

        return torch.FloatTensor(x), torch.LongTensor([direction])
