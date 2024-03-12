# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 23:29:15 2024

@author: anush
"""

import torch
from torch.utils.data import Dataset
import numpy as np


# class CustomDataset(Dataset):
#     def __init__(self, data, targets):
#         # Assuming data and targets are pandas DataFrames or Series
#         self.data = data
#         self.targets = targets
        
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         # Convert the data at a specific index to a tensor when requested
#         sample = torch.tensor(self.data.iloc[idx].values, dtype=torch.float32)
#         target = torch.tensor(self.targets.iloc[idx].values, dtype=torch.float32)
#         return sample, target



class CustomDataset(Dataset):
    def __init__(self, data, targets):
        # Assuming data and targets are numpy arrays
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert the data at a specific index to a tensor when requested
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return sample, target
