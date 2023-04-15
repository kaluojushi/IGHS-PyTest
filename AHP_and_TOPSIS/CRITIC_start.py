import numpy as np

m, n = 6, 4
Y = np.array([[0, 1, 0.8333, 1, 0.4, 0],
              [0.4167, 0.6, 1, 0, 0, 0],
              [1, 0, 0, 0.5, 1, 0],
              [0.6667, 0.4, 0.6667, 0.5, 0.8, 1]])

sigma = np.std(Y, axis=0, ddof=1)
print('sigma: ', sigma)
R = np.corrcoef(Y, rowvar=False)
print('R: ', R)
F = np.sum(1 - R, axis=0)
print('F: ', F)
C = sigma * F
print('C: ', C)
w = C / np.sum(C)
print('w: ', w)
