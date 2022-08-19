
import os
import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#dataset 

class CIFAR10:

    def __init__(self, batch_size):

        self.batch_size = batch_size

        self.train_dataset = datasets.CIFAR10(root="data", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))
        self.val_dataset = datasets.CIFAR10(root="data", train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))

    def train_dataloader(self):

        train_loader = DataLoader(self.train_dataset,
                              batch_size=self.batch_size,
                              shuffle=True,
                              pin_memory=True)

        train_var = np.var(self.train_dataset.data / 255.0)
        return train_loader, train_var


    def val_dataloader(self):

        val_loader = DataLoader(self.val_dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            pin_memory=True)
        return val_loader
