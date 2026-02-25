import numpy as np
import torch
from torch.utils.data import Dataset


class SFPDataset(Dataset):
    def __init__(self, scaled_data, actions, quality, tp_labels, sl_labels, window=30):
        self.data = scaled_data
        self.window = window

        # Only index SFP bars (where action != 0) that have enough history
        valid_start = window - 1
        self.indices = []
        for i in range(valid_start, len(actions)):
            if actions[i] != 0:
                self.indices.append(i)
        self.indices = np.array(self.indices)

        self.actions = actions
        self.quality = quality
        self.tp_labels = tp_labels
        self.sl_labels = sl_labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x = self.data[real_idx - self.window + 1 : real_idx + 1]
        direction = self.actions[real_idx]  # 1=long, 2=short
        q = self.quality[real_idx]
        tp = self.tp_labels[real_idx]
        sl = self.sl_labels[real_idx]
        return (
            torch.FloatTensor(x),
            torch.LongTensor([direction]).squeeze(),
            torch.FloatTensor([q]).squeeze(),
            torch.FloatTensor([tp]).squeeze(),
            torch.FloatTensor([sl]).squeeze(),
        )
