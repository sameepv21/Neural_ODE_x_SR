# Deep Learning Lib
import torch
import torch.nn as nn
import torch.optim as optim

# Data Manipulation Lib
import numpy as np
import random

# Data Loading Lib
import json

# Misc
import os
import warnings
warnings.filterwarnings('ignore')

# Define some constants
ROOT_PATH = "/Neural_ODE_x_SR/data"
FILE_NAME = 'strogatz_extended.json'
SEED = 42

# Define some default settings
torch.set_default_dtype(torch.float64)
decide = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Read dataset
with open(os.path.join(ROOT_PATH, FILE_NAME)) as strogatz:
    data = np.array(json.load(strogatz))

X_train = torch.tensor()