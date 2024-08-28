import os
import numpy as np
from torch.utils.data import DataLoader, Dataset, BatchSampler, Sampler

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        print(f"Process id {os.getpid()}, Index: {idx}\n")
        return self.data[idx]


class MyBatchSampler(BatchSampler):
    def __init__(self, data_source, batch_size, drop_last):
        super().__init__(data_source, batch_size, drop_last)
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.indices = list(range(len(self.data_source)))
        self.UsedIndices = set()
        self.BatchIndex = 0


    def __iter__(self):
        return self


    def __next__(self):
        if len(set(self.indices) - self.UsedIndices) == 0:
            raise StopIteration
        
        start_idx = self.BatchIndex*self.batch_size
        # indices = self.indices[start_idx:start_idx+self.batch_size]
        newIndices = list(set(self.indices)-self.UsedIndices)
        indices = np.random.choice(
            newIndices, self.batch_size,
            replace=len(newIndices)<=self.batch_size
        )

        self.UsedIndices.update(indices)

        self.BatchIndex += 1
        print(f"\033[34mBatch Sampler Indices: {indices}\033[0m", )
        return indices
            


dataset = MyDataset(list(range(50)))

batch_sampler = MyBatchSampler(dataset, batch_size=4, drop_last=False)

data_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)

for batch in data_loader:
    print(f"\033[01m\033[031m==> {batch}\033[0m")
