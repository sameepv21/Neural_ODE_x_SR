# Deep Learning Lib
import torch 
import torch.nn as nn

# Data Manipulation Lib
import numpy as np

def get_batch_single(data, t, batch_len=50, batch_size=50, device = torch.device("cpu")):
	s = torch.from_numpy(np.random.choice(np.arange(len(t) - batch_len, len(t), dtype=np.int64), batch_size, replace=False))

	batch_y0 = data[0,s]  # (M, D)
	batch_t = t[:batch_len]  # (T)
	batch_y = torch.stack([data[0,s]], dim=1)  # (T, M, D)
	
	return batch_y0.to(device), batch_t.to(device), batch_y.to(device)

def create_net(n_inputs, n_outputs, n_layers = 1, 
	n_units = 100, nonlinear = nn.Tanh):
	if n_layers == 0:
		layers = [nn.Linear(n_inputs, n_outputs)]
	else:
		layers = [nn.Linear(n_inputs, n_units)]
		for i in range(n_layers-1):
			layers.append(nonlinear())
			layers.append(nn.Linear(n_units, n_units))

		layers.append(nonlinear())
		layers.append(nn.Linear(n_units, n_outputs))
	return nn.Sequential(*layers)