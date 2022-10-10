# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import torch as t

from torch.utils.data import Dataset


class TsDataset(Dataset):

    def __init__(self,
                 data,
                 input_size,
                 output_size,
                 sample_freq=1):

        self.input_size = input_size
        self.output_size = output_size
        self.sample_freq = sample_freq

        # Create tensor
        self.n_series = data['unique_id'].nunique()

        self.data = np.zeros((self.n_series, 3, len(data)//self.n_series))
        self.data[:,0,:] = data['y'].values.reshape(self.n_series,-1)
        self.data[:,1,:] = data['available_mask'].values.reshape(self.n_series,-1)
        self.data[:,2,:] = data['sample_mask'].values.reshape(self.n_series,-1)

        self.data = t.Tensor(self.data)

        # Sampleable timestamps
        sampleable_ts = t.max(self.data[:, -1, :], dim=0)[0]
        self.first_ts = np.nonzero(sampleable_ts)[0,0]
        self.last_ts = np.nonzero(sampleable_ts)[-1,0]

        self.first_ts = int(self.first_ts)
        self.sampleable_stamps = self.last_ts - self.first_ts + 1
        self.sampleable_stamps = int(self.sampleable_stamps.cpu().detach().numpy())
        
    def __getitem__(self, idx: int):

        idx = self.sample_freq*idx

        if self.first_ts + 1 > self.input_size:
            idx = idx + self.first_ts
            if self.output_size > 0:
                idx = idx - self.input_size

        end_idx = idx + self.input_size + self.output_size
        Y = self.data[:, 0, idx:end_idx]

        available_mask = self.data[:, 1, idx:end_idx]
        sample_mask = self.data[:, 2, idx:end_idx]
        ts_idxs = t.as_tensor(np.arange(self.n_series), dtype=t.long)

        batch = {'Y': Y,
                 'X': t.Tensor([]),
                'available_mask': available_mask,
                'sample_mask': sample_mask,
                'ts_idxs': ts_idxs,
                'temporal_idxs': idx}

        return batch

    def __len__(self):
        if self.first_ts + 1 > self.input_size:
            return (self.sampleable_stamps - self.output_size + 1)//self.sample_freq
        else:
            return (self.sampleable_stamps - self.input_size - self.output_size + 1)//self.sample_freq
