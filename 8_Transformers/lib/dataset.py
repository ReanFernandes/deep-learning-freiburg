import numpy as np
from torch.utils.data import Dataset
import torch


class CountingDataset(Dataset):
    def __init__(self, vocab_size, length, dataset_size=64000):
        self.vocab_size = vocab_size
        self.length = length
        self.dataset_size = dataset_size
        vocab = np.random.choice(np.arange(vocab_size), [dataset_size, length])
        x = np.eye(vocab_size)[vocab]
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(np.sum(x, axis=1)).long()

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x[idx], self.y[idx]
