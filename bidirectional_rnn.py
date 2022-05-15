import torch
import torch.nn as nn
import numpy as np

class BidirectionalRNN(nn.Module):
    def __init__(self, D, H):
        super(VanillaRNN, self).__init__()
        self.Wxh = torch.normal(0, 1, (D, H))
        self.Whh = torch.normal(0, 1, (H, H))
        self.H = H

    def forward(self, x):
        h = torch.zeros(self.H)
        N, T, D = x.shape

        for t in range(T):
            x_i = x[:, t, :] # [N, D] vector
            h = torch.matmul(x_i, self.Wxh) + torch.matmul(h, self.Whh)
        return h

N = 1
T = 10
D = 20
H = 15

rnn = VanillaRNN(D, H)

X = torch.ones((N, T, D))
print(rnn(X))