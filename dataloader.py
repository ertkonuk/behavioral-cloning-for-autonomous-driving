# . . a simple DataLoader class for PyTorch that supports GPU
# . . using this loader makes sense only if the entire dataset will be copied to the GPU
# . . otherwise, use the PyTorch DataLoader: torch.utils.data.DataLoader
# . . DO NOT MOVE THE DATASET TO GPU BEFORE THE LOADER
# . . this will make a copy of the data set and use unnecessary GPU memory
# . . 
# . . assumes data were shuffled
# . . to do: add shuffling
# . . author: tugrul konuk
import numpy as np
import torch

class DataLoader:
    def __init__(self, dataset, batch_size=1, cuda=True):
        self.dataset = dataset

        # . . move to gpu
        if (cuda):
            self.dataset.x = self.dataset.x.cuda()
            if self.dataset.y is not None:
                self.dataset.y = self.dataset.y.cuda()
        
        if self.dataset.y is not None:
            self.datalength = len(dataset[:][0])
        else:
            self.datalength = len(dataset[:])
        
        self.batch_size=batch_size
        
        # . . full batch
        if self.batch_size == -1: 
            self.batch_size = self.datalength

        self.start = -self.batch_size
        self.end = len(dataset[:][0])-1
        

    def __iter__(self):
        self.start = -self.batch_size
        return self

    def __next__(self):
        self.start = self.start + self.batch_size
        if self.start <= self.datalength-1:
            self.end = self.start + self.batch_size            
            return self.dataset[self.start:self.end]
        else:
            raise StopIteration

    def __len__(self):
        return np.ceil(self.datalength/self.batch_size).astype(np.int32)