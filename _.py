import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class CustomBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=8):
        self.datasource = dataset
        self.batch_size = batch_size

    def __len__(self):
        return len(self.datasource)

    def __iter__(self):
        print("Batch Sampler PID:", os.getpid())
        indices = list(range(len(self.datasource)))
        for i in range(2):
            batchIndexes = np.random.choice(indices, self.batch_size)
            yield batchIndexes


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        print(f"OS PID: {os.getpid()}")
        return self.data[idx]


data = list(range(100, 200))  # Örnek veri
dataset = MyDataset(data)

batch_size = 8
customBatchSampler = CustomBatchSampler(dataset, batch_size=batch_size)
DATALOADER = DataLoader(
    dataset,
    batch_sampler=customBatchSampler,
    num_workers=1,
    persistent_workers=False, 
    pin_memory=True,
    multiprocessing_context = torch.multiprocessing.get_context("spawn")
    # collate_fn=custom_collate_fn,
)


if "__main__" == __name__:
    for i, batch in enumerate(DATALOADER):
        print(f"Batch: {i}, Data: {batch}")
        time.sleep(1)
