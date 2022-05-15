import torch
import torch.nn as nn
import numpy as np

class VanillaRNN(nn.Module):
    def __init__(self, D, H, O):
        super(VanillaRNN, self).__init__()
        self.Wxh = torch.normal(0, 1, (D, H))
        self.Whh = torch.normal(0, 1, (H, H))
        self.Who = torch.normal(0, 1, (H, O))
        self.H = H

    def forward(self, x):
        h = torch.zeros(self.H)
        N, T, D = x.shape

        for t in range(T):
            x_i = x[:, t, :] # [N, D] vector
            h = torch.matmul(x_i, self.Wxh) + torch.matmul(h, self.Whh)
        y_hat = torch.matmul(h, self.Who)
        return y_hat

N = 1
T = 10
D = 20
H = 15
O = 15

rnn = VanillaRNN(D, H, O)

X = torch.ones((N, T, D))
print(rnn(X))