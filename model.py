import torch
from torch import nn
import torch.nn.functional as F

# . . Convolutional Neural Network
# . . define the network architecture
class NvidiaNetwork(nn.Module):
    # . . the constructor
    def __init__(self):
        # . . call the constructor of the parent class
        super(NvidiaNetwork, self).__init__()

        # . . the network architecture
        # . . convolutional layerds for feature engineering
        self.conv2d = nn.Sequential(
                      nn.Conv2d(3, 24, 5, stride=2),
                      nn.ELU(),
                      nn.Conv2d(24, 36, 5, stride=2),
                      nn.ELU(),
                      nn.Conv2d(36, 48, 5, stride=2),
                      nn.ELU(),
                      nn.Conv2d(48, 64, 3),
                      nn.ELU(),
                      nn.Conv2d(64, 64, 3),
                      nn.Dropout(0.2)
                    )
        # . . fully connected layers for the steering prediction
        # nn.Linear(64 * 2 * 33, 100), . . track one
        # nn.Linear(64 * 4 * 33, 100), . . track two
        self.linear = nn.Sequential(
                      nn.Linear(64 * 2 * 33, 100),
                      nn.ELU(),
                      nn.Linear(100, 50),
                      nn.ELU(),
                      nn.Linear(50, 10),
                      nn.Linear(10, 1),
                      nn.Dropout(0.5)
                    )
        
    # . . forward propagation
    def forward(self, x):
        # . . reshape the input tensor                    
        #x = x.view(x.shape[0], 3, 64, 64) # . . track one 
        x = x.view(x.shape[0], 3, 70, 320) # . . track one
        #x = x.view(x.shape[0], 3, 90, 320) # . . track two 
        # . . convolutional layers
        x = self.conv2d(x)        
        # . . flatten the tensor for fully connected layers
        x = x.view(x.shape[0], -1)
        # . . fully connected layers
        x = self.linear(x)
        return x

class NetworkLight(nn.Module):

    def __init__(self):
        super(NetworkLight, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 48, 3, stride=2),
            nn.MaxPool2d(4, stride=4),
            nn.Dropout(p=0.25)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=48*4*19, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )
        

    def forward(self, x):
        #x = x.view(x.shape[0], 3, 64, 64) # . . track one
        x = x.view(x.shape[0], 3, 70, 320) # . . track one
        #x = x.view(x.shape[0], 3, 90, 320) # . . track two 
        x = self.conv_layers(x)
        #print(output.shape)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

class NetworkTK(nn.Module):

    def __init__(self):
        super(NetworkTK, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 48, 3, stride=2),
            nn.MaxPool2d(4, stride=4),
            nn.Dropout(p=0.25)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=48*4*19, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1),
            nn.Tanh()
        )
        

    def forward(self, x):
        #x = x.view(x.shape[0], 3, 64, 64) # . . track one
        x = x.view(x.shape[0], 3, 70, 320) # . . track one
        #x = x.view(x.shape[0], 3, 90, 320) # . . track two 
        x = self.conv_layers(x)
        #print(output.shape)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
