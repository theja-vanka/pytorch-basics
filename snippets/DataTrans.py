import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np


class TemplateDataSet(Dataset):
    def __init__(self, transform=None):
        # data loading
        dset = np.loadtxt(
                './data/wine.csv',
                delimiter=",",
                dtype=np.float32,
                skiprows=1
            )
        self.x = dset[:, 1:]
        # Since 1st column is target
        self.y = dset[:, [0]]  # n_samples, 1
        self.n_samples = dset.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        # len(dataset)
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets


dataset = TemplateDataSet(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = TemplateDataSet(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))
