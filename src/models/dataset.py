import numpy as np
import torch
from torch.utils.data import Dataset


class SFPDataset(Dataset):
    def __init__(self, scaled_data, actions, quality, mfe, sl_labels,
                 ttp=None, asset_ids=None, tf_ids=None, window=30):
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
        self.mfe = mfe
        self.sl_labels = sl_labels
        self.ttp = ttp if ttp is not None else np.zeros(len(actions), dtype=np.float32)
        self.asset_ids = asset_ids if asset_ids is not None else np.zeros(len(actions), dtype=np.int64)
        self.tf_ids = tf_ids if tf_ids is not None else np.zeros(len(actions), dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x = self.data[real_idx - self.window + 1 : real_idx + 1]
        direction = self.actions[real_idx]  # 1=long, 2=short
        q = self.quality[real_idx]
        mfe = self.mfe[real_idx]
        sl = self.sl_labels[real_idx]
        ttp = self.ttp[real_idx]
        asset_id = self.asset_ids[real_idx]
        tf_id = self.tf_ids[real_idx]
        return (
            torch.FloatTensor(x),
            torch.LongTensor([direction]).squeeze(),
            torch.FloatTensor([q]).squeeze(),    # scalar quality
            torch.FloatTensor([mfe]).squeeze(),   # scalar MFE
            torch.FloatTensor([sl]).squeeze(),    # scalar SL
            torch.FloatTensor([ttp]).squeeze(),   # scalar TTP
            torch.LongTensor([asset_id]).squeeze(),  # asset ID
            torch.LongTensor([tf_id]).squeeze(),     # timeframe ID
        )
