# Deep Learning Lib
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint

# Data Manipulation Lib
import numpy as np
import random

# Data Loading Lib
import json

# Helper Modules
from lib.utils import *
from lib.odefunc import *

# Misc
import os
import warnings
warnings.filterwarnings('ignore')

# Define some constants
ROOT_PATH = "./data"
CHECKPOINT_PATH = './checkpoint/state_dict_'
FILE_NAME = 'strogatz_extended.json'
SEED = 42
EPOCHS = 1000
N_LAYERS = 4
N_UNITS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define some default settings
torch.set_default_dtype(torch.float64)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Read dataset
with open(os.path.join(ROOT_PATH, FILE_NAME)) as strogatz:
    data = np.array(json.load(strogatz))

# Define Neural Networks for each of the ODE and each initial condition in the dataset
ode_funcs = []
for ode in data:
    for dim in range(len(ode['init'])):
        ode_func = ODEfunc(ode['dim'], N_LAYERS, N_UNITS, DEVICE)
        ode_funcs.append(ode_func)

# Set best loss for each ode to be -ve infinity
best_loss = [float('inf')] * len(ode_funcs)

# Make sure that there is no discrepancy
assert len(best_loss) == len(ode_funcs)

# Loop to train a single ode
from lib.utils import *

for index, neural_ode in enumerate(ode_funcs):
    id = index + 1 # id of the ode under consideration

    # Get the training data
    X_train_dict = data[index]['solutions'][0]

    # Number of trajectories 
    num_traj = int(data[index]['dim'])

    # Train for each trajectory
    for i in range(num_traj):
        # Define optimizer and scheduler
        optimizer = optim.Adamax(neural_ode.parameters())
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9987)
        
        # Training loop
        for itr in range(EPOCHS):

            # Separate required variables
            X_train = torch.tensor(X_train_dict[i]['y'])
            t = torch.tensor(X_train_dict[i]['t'])
            init_values = X_train_dict[i]['init']

            # Set required variables
            optimizer.zero_grad()
            
            # Get batches of data
            batch_y0, batch_t, batch_y = get_batch_single(X_train, t, device = DEVICE)
            batch_y0 = batch_y0.reshape(-1, 1)

            # Train the Neural ODEs
            pred_y = odeint(neural_ode, batch_y0, batch_t)
            loss = torch.mean((pred_y - batch_y)**2) * 10e4
            
            # Print the loss
            print(f'Epoch {itr} / {EPOCHS} Loss {loss.item()}')
            
            # Save checkpoint if lesser error
            if best_loss[index] > loss:
                best_loss[index] = loss
                ckpt_path = CHECKPOINT_PATH + str(id) + '.ckpt'
                # print(ckpt_path)
                torch.save({'state_dict': neural_ode.state_dict(),}, ckpt_path)

            # Backprop and step up
            loss.backward()
            optimizer.step()
            scheduler.step()