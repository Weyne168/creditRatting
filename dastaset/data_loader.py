import sys

sys.path.append('../')
import pickle

import torch
from torch.utils import data
import numpy as np


def load_data(data_pt):
    with open(data_pt, 'rb') as f:
        data = pickle.load(f)
    return data


class Dataset(data.Dataset):
    def __init__(self, dat_file, stride=6, repeat=10, is_testing=False):
        self.data = load_data(dat_file)
        self.is_testing = is_testing
        self.stride = stride
        self.repeat = repeat

        self._offsets = []
        for i in range(len(self.data)):
            d = self.data[i]
            seq_len = d.shape[0]
            if seq_len > self.stride:
                for k in range(seq_len - self.stride):
                    self._offsets.append((i, k))
            else:
                self._offsets.append((i, -1))
        self.dat_len = len(self._offsets)

    def __len__(self):
        if self.is_testing:
            return len(self._offsets)
        return len(self._offsets) * self.repeat

    def __getitem__(self, index):
        i = index % self.dat_len
        j, _offset = self._offsets[i]
        d = self.data[j]
        seq_len = d.shape[0]
        if seq_len > self.stride:
            sequence = d[_offset:_offset + self.stride]
        else:
            sequence = d

        # print(_offset, sequence.shape)
        label = sequence[-1, -1]
        X = sequence[:, :-1]
        X[:, -2] += 1  # the idx of category features should begin to 1, because 0 indicates information deficiency

        if _offset == -1:
            padding = np.zeros((self.stride - seq_len, X.shape[1]))
            X = np.concatenate([padding, X], axis=0)
        x_cls = X[:, -1]
        x_num = X[:, :-1]

        x_num = torch.from_numpy(x_num).float()
        x_cls = torch.from_numpy(x_cls).long()
        return x_num, x_cls.unsqueeze(-1), int(label)


def get_loader(dat_file, stride=3, repeat=10, batch_size=100, shuffle=True, num_workers=1, is_testing=False):
    """Builds and returns Dataloader."""
    dataset = Dataset(dat_file, stride, repeat, is_testing)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers
                                  )
    return data_loader


if __name__ == '__main__':
    pass
