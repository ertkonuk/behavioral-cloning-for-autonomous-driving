import torch
from torch import nn

# . . Physics-informed Neural Network
# . . define the network architecture
class NeuralNetwork(nn.Module):
    # . . the constructor
    def __init__(self, input_size, output_size, hidden_size, num_hidden, dropout=0.5):
        # . . call the constructor of the parent class
        super(NeuralNetwork, self).__init__()

        # . . get the num layers
        self.num_hidden = num_hidden

        # . . define the ANN structure
        # . . the input layer has three inputs: {t, z, x}
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        # . . hidden layers
        self.layers.extend([nn.Linear(hidden_size, hidden_size) for i in range(1, self.num_hidden-1)])
        # . . the output layer: {p: pressure}
        self.layers.append(nn.Linear(hidden_size, output_size))

        # . . dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # . . the activation function
        self.activation = nn.Tanh()

        ## . . Xavier initialization of weights
        for layer in self.layers:
            torch.nn.init.xavier_normal_(layer.weight)
            #torch.nn.init.xavier_uniform_(layer.weight)
            #torch.nn.init.normal_(layer.bias)
        
    # . . forward propagation
    def forward(self, x):
        ## . . flatten the input tensor         
        #x = x.view(x.shape[0], -1)
        
        # . . the input layer (no dropout & no activation)        
        x = self.activation(self.layers[0](x))

        # . . feed-forward the network (hidden layers)
        for layer in self.layers[1:-1]:
            #x = self.dropout(self.activation(layer(x)))
            x = self.activation(layer(x))

        # . . the output layer
        #x = self.activation(self.layers[-1](x))
        x = self.layers[-1](x)
        
        return x