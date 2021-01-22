import torch
from torch import optim
from torch import nn
from callbacks import *
import utils

# . . private utilities: do not call from the main program
from _utils import _find_optimizer

# . . experimental functionality: mixed precision training with Nvidia Apex 
# . . and automatic mixed precision (amp)
from apex import amp


class Trainer(object):
    def __init__(self, model, device="cuda"):
        # . . the constructor
        # . . set the model
        self.model = model

        # . . set the device
        self.device = device

        # . . if already not, copy to gpu 
        self.model = self.model.to(self.device)

        # . . callback functions
        self.callbacks = []

        # . . other properties
        self.model._stop_training = False

    # . . sets the optimizer and callbacks
    def compile(self, optimizer='adam', criterion=None,  callbacks=None, **optimargs):    
            # . . find the optimizer in the torch.optim directory   
            optimizer = _find_optimizer(optimizer)
            self.optimizer = optimizer(self.model.parameters(), **optimargs)
            
            # . . default callbacks
            # . . epoch-level statistics
            self.callbacks.append(EpochMetrics(monitor='loss', skip=1))
            # . . batch-level statistics
            #self.callbacks.append(BatchMetrics(monitor='batch_loss', skip=1))
            
            #  . . the user-defined callbacks
            if callbacks is not None:   self.callbacks.extend(callbacks)
            
            # . . set the loss function
            if criterion is None:
                self.criterion = nn.MSELoss()
            else:
                self.criterion = criterion

            # . . initialize the mixed precision training
            self.model, self.optim_pde = amp.initialize(self.model, self.optimizer, opt_level="O0")
    # . . use if you want to change the optimizer 
    # . . to do: implement
    def recompile(self, optimizer, callbacks=None): pass
    
     
    def fit(self, trainloader, validloader, num_epochs=1000):        

        # . . logs
        logs = {}

        # . . number of batches
        num_train_batch = len(trainloader)
        num_valid_batch = len(validloader)

        # . . register num batches to logs
        logs['num_train_batch'] = num_train_batch
        logs['num_valid_batch'] = num_valid_batch

        # . . 
        # . . set the callback handler
        callback_handler = CallbackHandler(self.callbacks)

        # . . keep track of the losses        
        train_losses = []
        valid_losses = []


        # . . call the callback function on_train_begin(): load the best model
        callback_handler.on_train_begin(logs=logs, model=self.model)

        for epoch in range(num_epochs):
            #scheduler.step()
            train_loss = 0.
            valid_loss = 0.
          
            # . . activate the training mode
            self.model.train()

            # . . get the next batch of training data
            for batch, (center, left, right) in enumerate(trainloader):                                

                # . . the training loss for the current batch
                batch_loss = 0.

                # . . send the batch to GPU
                center, left, right = utils.to(center, self.device), utils.to(left, self.device), utils.to(right, self.device)                

                # . . zero the parameter gradients
                self.optimizer.zero_grad()

                # . . iterate over cameras
                cameras = [center, left, right]                
                for camera in cameras:
                    image, steering = camera
                    
                    # . . feed-forward network to predict the steering angle
                    angle = self.model(image)

                    #log_angle = torch.log10(1.0 + angle)

                    # . . calculate the loss function
                    loss = self.criterion(angle, steering.unsqueeze(1))

                    # . . backpropagate the loss
                    #loss.backward()
                    # . . backpropogate the scaled physics loss
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()

                    # . . update weights
                    self.optimizer.step()
                    
                    # . . training loss for the current batch: accumulate over cameras
                    batch_loss += loss.item()

                    # . . accumulate the training loss
                    train_loss += loss.item()

                # . . register the batch training loss
                logs['batch_loss'] = batch_loss

                # . . call the callback functions on_epoch_end()                
                callback_handler.on_batch_end(batch, logs=logs, model=self.model)

            # . . activate the evaluation (validation) mode
            self.model.eval()
            # . . turn off the gradient for performance
            with torch.set_grad_enabled(False):
                # . . get the next batch of validation data
                for batch, (center, left, right) in enumerate(validloader): 

                    # . . send the batch to GPU
                    center, left, right = utils.to(center, self.device), utils.to(left, self.device), utils.to(right, self.device)                    
                
                    # . . iterate over cameras
                    cameras = [center, left, right]
                    for camera in cameras:
                        image, steering = camera

                        # . . feed-forward network to predict the steering angle
                        angle = self.model(image)

                        # . . calculate the loss function
                        loss = self.criterion(angle, steering.unsqueeze(1))                

                        # . . accumulate the training loss
                        valid_loss += loss.item()

            # . . normalize the training and validation losses
            train_loss /= (num_train_batch*3)
            valid_loss /= (num_valid_batch*3)

            # . . on epoch end
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            # . . update the epoch statistics (logs)
            logs["train_loss"] = train_loss
            logs["valid_loss"] = valid_loss

            # . . call the callback functions on_epoch_end()                
            callback_handler.on_epoch_end(epoch, logs=logs, model=self.model)
    
            # . . check if the training should continue
            if self.model._stop_training:
                break

        # . . call the callback function on_train_end(): load the best model
        callback_handler.on_train_end(logs=logs, model=self.model)

        return train_losses, valid_losses
 