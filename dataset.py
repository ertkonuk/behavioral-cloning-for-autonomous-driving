# . . import torch Dataset class
from torch.utils.data import Dataset
# . . import the utilities
import utils
# . . numpy
import numpy as np

# . .  the dataset class inherits from torch
class Dataset(Dataset):

    # . . constructor 

    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        # . . get one item from the dataset        
        # . . steering angle
        steering = float(self.target[index])
        
        # . . images from three cameras
        cameras = self.data[index]
        center  = cameras[0]
        left    = cameras[1]
        right   = cameras[2]

        # . . size of the processed image for the training: (width, height)
        imsize = (70,320)
        # . . probability of fliping an image
        pflip = 0.5
        # . . correct steering for left and right directions
        steering_correction = 0.25
        # . . read and correct the steering for the left and right cameras 
        img_center, steering_center = utils.augment(center,steering, imsize, pflip)
        img_left  , steering_left   = utils.augment(left  ,steering + steering_correction, imsize, pflip)
        img_right , steering_right  = utils.augment(right ,steering - steering_correction, imsize, pflip)
        
        # . . augment steering: there is not a single steering value that works. so add random noise to augment it
        steering_augmentation = 0.25
        # . . add zero mean Gaussian noise        
        steering_left   += np.random.normal(loc=0, scale=steering_augmentation)
        steering_center += np.random.normal(loc=0, scale=steering_augmentation)
        steering_right  += np.random.normal(loc=0, scale=steering_augmentation)

        if self.transform is not None:
            # . . scale images to the range between zero and one
            img_center = self.transform(img_center)
            img_left   = self.transform(img_left)
            img_right  = self.transform(img_right)
        
        return (img_center, steering_center), (img_left, steering_left), (img_right, steering_right)


    def __len__(self):
        return len(self.data)