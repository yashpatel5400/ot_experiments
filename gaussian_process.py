import math
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt

class ExactGPModel(gpytorch.models.ExactGP):
	def __init__(self, train_x, train_y, likelihood):
		super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
		self.mean_module = gpytorch.means.ConstantMean()
		self.cov_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

	def forward(self, x):
		mean_x = self.mean_module(x)
		cov_x = self.cov_module(x)
		return gpytorch.distributions.MultivariateNormal(mean_x, cov_x)

train_x = torch.linspace(0, 1, 100)
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(.04)

plt.plot(train_x, train_y)
plt.show()

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

likelihood.train()
model.train()