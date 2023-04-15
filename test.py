import numpy as np
from matplotlib import pyplot as plt

import MZJ_GMOMPA.objective as obj
from MZJ_GMOMPA.test_data import test_data as td

u = td.values()

ul = np.array([3, 90.45, 732, 2.6])
ll = np.array([2, 75.05, 609, 1.2])
pop = np.zeros((1000, 7))
for i in range(1000):
    for j in range(4):
        pop[i, j] = np.random.uniform(ll[j], ul[j])

pop[:, 0:3] = np.round(pop[:, 0:3])
pop[:, 3] = np.round(pop[:, 3] * 100) / 100
for i in range(1000):
    pop[i, 4:] = obj.fitness(pop[i, :4], u)
energy = pop[:, 4][np.argsort(pop[:, 4])]
plt.scatter(range(len(energy)), energy, marker='.')
plt.show()
time = pop[:, 5][np.argsort(pop[:, 5])]
plt.scatter(range(len(time)), time, marker='.')
plt.show()
quality = pop[:, 6][np.argsort(pop[:, 6])]
plt.scatter(range(len(quality)), quality, marker='.')
plt.show()
