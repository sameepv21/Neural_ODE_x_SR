import torch
from lib.utils import *

class ODEfunc(nn.Module):
	def __init__(self, dim, nlayer, nunit, device = torch.device("cpu")):
		super(ODEfunc, self).__init__()
		self.gradient_net = create_net(dim, dim, n_layers=nlayer, n_units=nunit, nonlinear = nn.Tanh).to(device)
		self.NFE = 0

	def forward(self, t, y):
		output = self.gradient_net(y)
		return output