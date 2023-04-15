import numpy as np


m = 3
order = np.array([2, 0, 1])
k = np.array([5, 3])
ratio = np.zeros(m - 1)
for i in range(m - 1):
    ratio[i] = 9 / (10 - k[i])
print(ratio)
weight = np.zeros(m)
x = 0
for i in range(m - 1):
    y = 1
    for j in range(i, m - 1):
        y *= ratio[j]
    x += y
weight[order[-1]] = 1 / (1 + x)
for i in range(m - 2, -1, -1):
    weight[order[i]] = weight[order[i + 1]] * ratio[i]
print(weight)
print(np.sum(weight))
