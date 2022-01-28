import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
import matplotlib.pyplot as plt

n = 100  # nb bins

# bin positions
x = np.arange(n, dtype=np.float64)

# Gaussian distributions
# a = np.random.normal(20, 5, 1000)
# a_hist, bins_edges = np.histogram(a, bins=30)
# a_prob = a_hist / np.linalg.norm(a_hist)
# bins = [(bins_edges[i] + bins_edges[i+1]) / 2 for i in range(len(bins_edges) - 1)]

a = gauss(n, m=20, s=5)  # m= mean, s= std
b = gauss(n, m=60, s=10)

print(x)

# loss matrix
M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))
M /= M.max()

# plt.plot(bins, a_prob) # true notion of empirical PDF
# plt.plot(x, b)
# plt.show()

G0 = ot.emd(a, b, M)

pl.figure(3, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, G0, 'OT matrix G0')
plt.show


lambd = 1e-3
Gs = ot.sinkhorn(a, b, M, lambd, verbose=True)

pl.figure(4, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, Gs, 'OT matrix Sinkhorn')

pl.show()
