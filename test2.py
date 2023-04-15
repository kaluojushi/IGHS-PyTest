import numpy as np

w1 = np.random.uniform(0, 1)
print('w1: ', w1)
w2 = np.random.uniform(0, 1 - w1)
print('w2: ', w2)
w3 = np.random.uniform(0, 1 - w1 - w2)
print('w3: ', w3)
w4 = np.random.uniform(0, 1 - w1 - w2 - w3)
print('w4: ', w4)
w5 = 1 - w1 - w2 - w3 - w4
print('w5: ', w5)
