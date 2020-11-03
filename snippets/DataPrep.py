import torch
# import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math


class TemplateDataSet(Dataset):

    def __init__(self):
        # data loading
        dset = np.loadtxt(
                './data/wine.csv',
                delimiter=",",
                dtype=np.float32,
                skiprows=1
            )
        self.x = torch.from_numpy(dset[:, 1:])
        # Since 1st column is target
        self.y = torch.from_numpy(dset[:, [0]])  # n_samples, 1
        self.n_samples = dset.shape[0]

    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples


dataset = TemplateDataSet()
dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=True,
        num_workers=3
    )

# training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward backward, update
        if (i+1) % 5 == 0:
            print(f'epoch: {epoch+1}/{num_epochs}', end=", ")
            print(f'step: {i+1}/{n_iterations}', end=", ")
            print(f'inputs: {inputs.shape}')

# torchvision.datasets.MNIST()
