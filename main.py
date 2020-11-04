# . . import libraries
import os
from pathlib import Path
# . . pytorch modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# . . pandas 
import pandas as pd
# . . numpy
import numpy as np
# . . scikit-learn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
# . . matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as npimg
# . .  set this to be able to see the figure axis labels in a dark theme
from matplotlib import style
#style.use('dark_background')
# . . to see the available options
# print(plt.style.available)
# . . import opencv
import cv2
# . . scipy
import scipy
from scipy import signal
# . . import libraries by tugrulkonuk
import utils
from utils import parse_args
from dataset import Dataset
from model import *
from trainer import Trainer
from callbacks import ReturnBestModel, EarlyStopping

# . . parse the command-line arguments
args = parse_args()

# . . set the device
if torch.cuda.is_available():  
    device = torch.device("cuda")  
else:  
    device = torch.device("cpu")      

# . . set the default precision
dtype = torch.float32

# . . use cudnn backend for performance
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# . . parameters
# . . user-defined
num_epochs    = args.epochs
batch_size    = args.batch_size
learning_rate = args.lr
train_size    = args.train_size
min_delta     = args.min_delta
patience      = args.patience 
num_workers   = args.num_workers
pin_memory    = args.pin_memory
# . . computed
test_size     = 1.0 - train_size

# . . debug
# . . display images?
display = True

# . . import the data
# . . define the data directory and file name
datadir = args.datapath
datafile= 'driving_log.csv'
# . . three camera images and the driving parameters
header = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
data = pd.read_csv(os.path.join(datadir,datafile), names=header)

# . . remove the header from the data
data = data.iloc[1:]

# . . display first a few rows
pd.set_option('display.max_colwidth', -1)
data.head()

# . . data processing
# . . strip the path names from the camera images since they all have the same path
data['center'] = data['center'].apply(utils.strip_path)
data['left'] = data['left'].apply(utils.strip_path)
data['right'] = data['right'].apply(utils.strip_path)
data.head()


# . . from pandas dataframe to numpy array
img_paths = data[['center','left','right']].to_numpy()
steerings = data[['steering']].to_numpy().astype(np.float32)

if display:
    # . . plot the histogram of \data
    angle = steerings
    # . . plot the steering wheel angle
    num_bins = 50
    samples_per_bin = 10
    hist, bin_edges = np.histogram(steerings, num_bins)
    # . . center bins around zero
    bins = bin_edges[:-1] + bin_edges[1:] * 0.5
    # . . plot the steering histogram
    plt.bar(bins, hist, width=0.05)
    plt.plot((np.min(data['steering']), np.max(data['steering']), samples_per_bin))
    plt.xlim(-1.0,1.0)
    plt.show()

    # . . plot images 
    frm = 50
    cpath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) +'/data/IMG/' + data['center'].iloc[frm].split('/')[-1]
    lpath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) +'/data/IMG/' + data['left'].iloc[frm].split('/')[-1]
    rpath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) +'/data/IMG/' + data['right'].iloc[frm].split('/')[-1]
    center = cv2.imread(cpath)
    left = cv2.imread(lpath)
    right = cv2.imread(rpath)
    plt.subplots(figsize=(20, 10))
    plt.subplot(131),plt.imshow(left); plt.axis("off")
    plt.subplot(132),plt.imshow(center); plt.axis("off")
    plt.subplot(133),plt.imshow(right); plt.axis("off")


    # . . plot images after cropping
    ccenter = center[65:-25, :, :]
    cleft = left[65:-25, :, :]
    cright = right[65:-25, :, :]
    print(ccenter.shape)
    plt.subplots(figsize=(20, 10))
    plt.subplot(131),plt.imshow(cleft); plt.axis("off")
    plt.subplot(132),plt.imshow(ccenter); plt.axis("off")
    plt.subplot(133),plt.imshow(cright); plt.axis("off")

    # . . steering angles before and after smoothing
    #plt.plot(steerings,'b')
    #plt.plot(signal.savgol_filter(steerings.reshape(1,-1), 55, 11).reshape(-1,1),'r')

# . . smooth the steering angle data using the Savitzky-Golay filter
#steerings = signal.savgol_filter(steerings.reshape(1,-1), 55, 11).reshape(-1,1)

# . . split data to training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(img_paths, steerings, test_size=test_size)

# . . define the data transformations
transformations = transforms.Compose([transforms.Lambda(lambda x: (x / 255.0) - 0.5)])

# . . torch dataset
train_data = Dataset(X_train, y_train, transformations)
valid_data = Dataset(X_valid, y_valid, transformations)

# . . train and validation data loaders
trainloader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
validloader = DataLoader(dataset=valid_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

# . . define the model
#model = NvidiaNetwork()
model = NetworkLight()

# . . create the trainer
trainer = Trainer(model, device)

# . . compile the trainer
# . . define the loss
criterion = nn.MSELoss()
# . . define the optimizer
optimparams = {'lr':learning_rate
              }
# . . define the callbacks
cb=[ReturnBestModel(), EarlyStopping(min_delta=min_delta, patience=patience)]
trainer.compile(optimizer='adam', callbacks=cb, **optimparams)

# . . train the network
train_loss, valid_loss = trainer.fit(trainloader, validloader, num_epochs=num_epochs)

# . . plot the loss
plt.plot(train_loss)
plt.plot(valid_loss)
plt.legend(['train_loss', 'valid_loss'])
plt.show()


# . . save the model
state = {
        'model': trainer.model.module if device == 'cuda' else trainer.model,
        }

torch.save(state, 'model.h5')