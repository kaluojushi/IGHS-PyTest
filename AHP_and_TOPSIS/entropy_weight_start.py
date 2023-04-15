import numpy as np

m, n = 5, 4
Y = np.array([[0, 1, 0.8333, 1, 0.4, 0],
              [0.4167, 0.6, 1, 0, 0, 0],
              [1, 0, 0, 0.5, 1, 0],
              [0.6667, 0.4, 0.6667, 0.5, 0.8, 1]])

G = Y / np.sum(Y, axis=0)
print('G: ', G)
G_ = G * np.log(G)
print('G_: ', G_)
G_ = np.nan_to_num(G_)
print('G_: ', G_)
E = np.sum(G_, axis=0) / -np.log(n)
print('E: ', E)
w = (1 - E) / np.sum(1 - E)
print('w: ', w)
