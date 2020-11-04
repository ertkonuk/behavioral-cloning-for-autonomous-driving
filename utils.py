# . . GPUtil for memory management
import platform
import numpy as np
import ntpath
import os
# . . opencv
import cv2
import random
import argparse

# . . do not import on MacOSX
if platform.system() is not 'Darwin':
    import GPUtil
# . . parse the command line parameters
def parse_args():

    parser = argparse.ArgumentParser(description='physics informed neural networks for 2D AWE solutions.')   
    # . . data directory
    default_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) +'/data/'
    parser.add_argument('--datapath',     type=str,   default=default_dir, help='data path')
    # . . hyperparameters
    parser.add_argument('--lr',           type=float, default=5e-4,  help='learning rate')
    parser.add_argument('--batch_size',   type=int,   default=32,    help='training batch size')
    parser.add_argument('--train_size',   type=float, default=0.8,   help='fraction of grid points to train')

    # . . training parameters
    parser.add_argument('--epochs',       type=int,   default=100,   help='number of epochs to train')

    # . . parameters for early stopping
    parser.add_argument('--patience',     type=int,   default=5,      help='number of epochs to wait for improvement')
    parser.add_argument('--min_delta',    type=float, default=0.001,  help='min loss function reduction to consider as improvement')
    
    # . . parameters for data loaders
    parser.add_argument('--num_workers',    type=int,  default=0,      help='number of workers to use inf data loader')
    parser.add_argument('--pin_memory' ,    type=bool, default=False,  help='use pin memory for faster cpu-to-gpu transfer')

    # . . parse the arguments
    args = parser.parse_args()

    return args


def gpuinfo(msg=""):
    print("------------------")
    print(msg)
    print("------------------")
    GPUtil.showUtilization()
    print("------------------")

def devinfo(device):
    print("------------------")
    print("torch.device: ", device)
    print("------------------")
   
def batchinfo(loader, label=True):

    print("------------------")
    print("There are {} batches in the dataset".format(len(loader)))
    if label:
        for x, y in loader:
            print("For one iteration (batch), there are:")
            print("Data:    {}".format(x.shape))
            print("Label:   {}".format(y.shape))
            break   
    else:
        for x in loader:
            print("For one iteration (batch), there are:")
            print("Data:    {}".format(x.shape))
            break  
    print("------------------")

def conv_output_shape(h_w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    taken from: https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5
    """
    
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    
    if type(stride) is not tuple:
        stride = (stride, stride)
    
    if type(padding) is not tuple:
        padding = (padding, padding)
    
    h = (h_w[0] + (2 * padding[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
    w = (h_w[1] + (2 * padding[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1
    
    return h, w


# . . strips the path name off from a full path filename
def strip_path(path):
  head, tail = ntpath.split(path)
  return tail


# . . augments the data for NN training
def augment(frame, angle, w_h, pflip=0.5):
    filename = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) +'/data/IMG/' + frame.split('/')[-1]
    
    # . . read the current frame(image)
    image = cv2.imread(filename)
    # . . crop the image to remove unrelated features
    image = image[65:-25, :, :] # . . track one
    #image = image[45:-25, :, :] # . . track two
   
    ## . . change image form RGB to YUV
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    # . . change image form BGR to HSV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    ## . .  augment the brightness of the image
    #min_brightness_augmentation = 0.2
    #max_brightness_augmentation = 1.5
    ## . .multiply the Value channel woth a random number
    #image[:, :, 2] *= random.uniform(min_brightness_augmentation, max_brightness_augmentation)
    #image[:, :, 2] = np.clip(image[:, :, 2], a_min=0, a_max=255)
    ## . . back to BGR
    #image = cv2.cvtColor(frame, code=cv2.COLOR_HSV2BGR)

    # . . apply Gaussian blur
    image = cv2.GaussianBlur(image, (3,3), 0)

    # . . augment the data
    # . . flip horizontally to generalize the images for both directions
    # . . this should help keeping the car in the middle of the road
    if np.random.rand() < pflip:
        image = cv2.flip(image, 1)
        # . . flip the steering angle to flip the direction of the turn
        angle = -1.0 * angle

    # . . resize the image 
    image = cv2.resize(image, w_h).astype(np.float32)

    return image, angle


# . . move a list of tensor to cuda (device)
def to(camera, device='cuda'):
    # . . separate images and steering angles
    images, angles = camera
    # . . move everything to device and return a list
    return images.float().to(device), angles.float().to(device)
