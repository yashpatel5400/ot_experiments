import torch
import torch.nn as nn
import numpy as np

class BidirectionalRNN(nn.Module):
    def __init__(self, D, H, O):
        super(BidirectionalRNN, self).__init__()
        # forward half of RNN
        self.forward_Wxh = torch.normal(0, 1, (D, H))
        self.forward_Whh = torch.normal(0, 1, (H, H))

        # backward half of RNN
        self.backward_Wxh = torch.normal(0, 1, (D, H))
        self.backward_Whh = torch.normal(0, 1, (H, H))

        self.Who = torch.normal(0, 1, (H, O))
        self.H = H

    def forward(self, x):
        N, T, D = x.shape

        forward_h = torch.zeros(self.H)
        backward_h = torch.zeros(self.H)

        for t in range(T):
            forward_x_i = x[:, t, :] # [N, D] vector
            forward_h = torch.matmul(forward_x_i, self.forward_Wxh) + torch.matmul(forward_h, self.forward_Whh)

            backward_x_i = x[:, -t, :] # [N, D] vector
            backward_h = torch.matmul(backward_x_i, self.backward_Wxh) + torch.matmul(backward_h, self.backward_Whh)

        h = forward_h + backward_h
        y_hat = torch.matmul(h, self.Who)
        return y_hat

N = 1
T = 10
D = 20
H = 15
O = 10

rnn = BidirectionalRNN(D, H, O)

X = torch.ones((N, T, D))
print(rnn(X))