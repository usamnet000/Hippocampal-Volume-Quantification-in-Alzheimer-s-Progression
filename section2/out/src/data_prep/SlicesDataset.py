"""
Module for Pytorch dataset representations
"""

import torch
from torch.utils.data import Dataset

class SlicesDataset(Dataset):
    """
    This class represents an indexable Torch dataset
    which could be consumed by the PyTorch DataLoader class
    """
    def __init__(self, data):
        self.data = data

        self.slices = []

        for i, d in enumerate(data):
            for j in range(d["image"].shape[0]):
                self.slices.append((i, j))

    def __getitem__(self, idx):
        """
        This method is called by PyTorch DataLoader class to return a sample with id idx

        Arguments: 
            idx {int} -- id of sample

        Returns:
            Dictionary of 2 Torch Tensors of dimensions [1, W, H]
        """
        slc = self.slices[idx]
        sample = dict()
        sample["id"] = idx
        
        # create two new keys in the "sample" dictionary, "image" and "seg"
        i, j = slc
        sample['image'] = torch.from_numpy(self.data[i]['image'][j, :, :][None, :])
        sample['seg'] = torch.from_numpy(self.data[i]['seg'][j, :, :][None, :])

        return sample

    def __len__(self):
        """
        This method is called by PyTorch DataLoader class to return number of samples in the dataset

        Returns:
            int
        """
        return len(self.slices)
