#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset


class BldgDataset(Dataset):
    def __init__(self, data_df):
        data_df.reset_index(inplace=True, drop=True)
        self.data = data_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = torch.tensor(self.data.iloc[idx, :-1], dtype=torch.float32)
        y = torch.tensor(self.data.iloc[idx, -1])

        return X, y

    def num_feats(self):
        return len(self.data.columns) - 1

    def num_classes(self):
        return self.data[self.data.columns[-1]].max() + 1
