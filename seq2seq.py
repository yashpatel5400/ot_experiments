import torch
import torch.nn as nn
import numpy as np

class Seq2Seq(nn.Module):
    def __init__(self, D, H, O):
        super(Seq2Seq, self).__init__()
        self.H = H
        self.D = D
        self.O = O

        # --- encoder --- #
        self.enc_Wxh = torch.normal(0, 1, (self.D, self.H))
        self.enc_Whh = torch.normal(0, 1, (self.H, self.H))

        # --- decoder --- #
        self.dec_Wxh = torch.normal(0, 1, (self.D, self.H))
        self.dec_Whh = torch.normal(0, 1, (self.H, self.H))
        self.dec_Who = torch.normal(0, 1, (self.H, self.O))

    def forward(self, x):
        N, T, D = x.shape

        # --- encoder --- #
        h = torch.zeros(self.H)
        for t in range(T):
            x_i = x[:, t, :]
            h = torch.sigmoid(torch.matmul(x_i, self.enc_Wxh) + torch.matmul(h, self.enc_Whh))

        # --- decoder --- #
        y = []
        s = h # feed in encoded context to initialize decoding side
        y_i = torch.normal(0, 1, (1, self.D)) # would normally be start token

        for t in range(T): # normally go until end token
            s = torch.sigmoid(torch.matmul(y_i, self.dec_Wxh) + torch.matmul(s, self.dec_Whh))
            y_i = torch.matmul(s, self.dec_Who)
            y.append(y_i)

        y_hat = torch.stack(y)
        return y_hat

N = 1
T = 10
H = 15
D = 10
O = 10

X = torch.normal(0, 1, (N, T, D))
seq2seq = Seq2Seq(D, H, O)
print(seq2seq(X).shape)