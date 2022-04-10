import math
import torch
import numpy as np
import matplotlib.pyplot as plt

train_x = torch.linspace(0, 1, 100)
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(.04)

plt.plot(train_x, train_y)
plt.show()