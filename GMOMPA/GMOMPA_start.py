import time

import numpy as np
from matplotlib import pyplot as plt

from pareto_lib import *
from predator_lib import *
from prey_movement_lib import *
from print_lib import *

dtype = np.dtype([('p', np.float64, 4), ('f', np.float64, 3)])

Hob = np.array([[1, 70.25], [1, 75.5], [1, 78], [1, 80.5], [1, 83.5], [1, 85.5], [1, 89], [1, 90.2],
                [2, 70.25], [2, 75.5], [2, 78], [2, 80.5], [2, 83.5], [2, 85.5], [2, 89], [2, 90.2],
                [3, 70.25], [3, 75.5], [3, 78], [3, 80.5], [3, 83.5], [3, 85.5], [3, 89], [3, 90.2]],
               dtype=np.float64)

# Determine the search space
# 确定搜索空间
# upper_limit = np.array([3, 80, 410, 1.6])
# lower_limit = np.array([2, 75, 375, 1.5])
upper_limit = np.array([2, 85, 720, 2.6], dtype=np.float64)
lower_limit = np.array([2, 80, 680, 2], dtype=np.float64)
# Expand the search space
# 扩大搜索空间
u = 0.95
for i in [1, 2]:
    y, x = upper_limit[i], lower_limit[i]
    lower_limit[i] = u * x
    upper_limit[i] = y + (1 - u) * x
# Make sure that the search space meets the accuracy requirements
# 确保搜索空间满足精度要求
for i in [0, 2]:
    lower_limit[i] = np.ceil(lower_limit[i])
    upper_limit[i] = np.floor(upper_limit[i])
for i in [1, 3]:
    lower_limit[i] = np.ceil(lower_limit[i] * 100) / 100
    upper_limit[i] = np.floor(upper_limit[i] * 100) / 100
# Add a small value to the upper limit to make sure that the upper limit is included in the search space
# 为上限添加一个小值，以确保上限包含在搜索空间中
upper_limit += 1e-5
print(upper_limit, lower_limit)  # [2, 89, 754, 2.6] [2, 76, 646, 2]


# Fitness function
# 适应度函数
def fitness(p):
    z0, da0, n0, f = p
    # mn, alpha, z1, beta, d1, B, h = 2, 0.349, 41, 0.456, 94.011, 13, 71
    mn, alpha, z1, beta, d1, B, h = 2.5, 0.349, 45, 0.297, 128.63, 45, 8.45
    Ps, ts, Psc, kappa1, kappa2 = 2200, 5, 200, -0.078, 2e-6
    Lr, vr, La, Ein, Uout = 104.5, 1500, 21.168, 2, 2
    epsilon1, epsilon2 = 0.035, 1.3e-5
    C, K1, K2, K3, X, Y, Z, U, V = 18.2, 1, 1.05, 1.11, 1.75, 0.65, 0.81, 0.26, 0.27
    omega1, omega2, zk = 0.5, 0.5, 17
    ta = La * z1 / (z0 * n0 * f) + Lr / vr
    tc = z1 * (Ein + B + Uout) / (z0 * n0 * f)
    Pr = C * K1 * K2 * K3 * (mn ** X) * (f ** Y) * (h ** Z) * ((np.pi * da0 * n0 / 1000) ** (1 - U)) * (z1 ** V) / da0
    energy = (Ps * ts + (Ps + Psc + kappa1 * n0 + kappa2 * n0 * n0) * (ta + tc) + (
            (1 + epsilon1) * Pr + epsilon2 * Pr * Pr) * tc) / 6e4
    times = (ts + ta + tc) * 60
    quality = omega1 * f * f * np.sin(alpha) / (4 * da0) + omega2 * np.pi * np.pi * mn * z0 * z0 * np.sin(alpha) / (
            4 * z1 * zk * zk)
    return np.array([energy, times, quality])


def build_prey(N):
    Prey = np.zeros((N, 4))
    for i in range(N):
        for j in range(4):
            Prey[i, j] = np.random.uniform(lower_limit[j], upper_limit[j])
    return Prey


start_time = time.time()
N = 100
max_arch = 100

iter = -1
max_iter = 100
Prey = None
# print(population)
# print(Prey)
Predator = None
Archive = None
last_Prey = None
last_Archive = None
while iter < max_iter:
    print(color('********** iter: %d **********', 'y') % iter)
    # If it is the initial iteration, build the prey population
    # Otherwise, prey population moves
    # 如果是最初迭代，构建 Prey 种群；否则，Prey 种群移动
    if iter == -1:
        Prey = build_prey(N)
        Prey = attraction(Prey, N, X_max=upper_limit, X_min=lower_limit, Hob=Hob)
        Apop = Prey.copy()
    else:
        # elite = Predator[np.random.randint(0, len(Predator))]
        # Elite = np.array([elite for _ in range(N)])
        Prey = update_prey(Prey, Predator, N, X_max=upper_limit, X_min=lower_limit, iter=iter, max_iter=max_iter,
                           Hob=Hob)
        # print('%d\' Prey after update: ' % iter, Prey)
        Prey = update_prey(Prey, Predator, N, X_max=upper_limit, X_min=lower_limit, iter=iter, max_iter=max_iter,
                           Hob=Hob, use_FADs=True)
        # print('%d\' Prey after FADs: ' % iter, Prey)
        Apop = construct_Apop(Prey, last_Prey, last_Archive)
    # After getting Apop, sort it, find Predator population, and update Archive
    # 得到 Apop 后，对其进行排序，找到 Predator 种群，更新 Archive
    sorted_layers = get_sorted_front_layers(Apop, use_fitness=True, fitness=fitness)
    Apop = get_sorted_new_population(Apop, sorted_layers, use_fitness=True, fitness=fitness)
    print(color('%d\'', 'g') % iter, 'Apop: %d' % len(Apop))
    Predator = construct_predator_population(Apop, len(sorted_layers[0]), N)
    print(color('%d\'', 'g') % iter, 'Predator: %d' % len(Predator))
    if iter == -1:
        Archive = Apop[sorted_layers[0]]
    else:
        Archive = update_archive_population(Archive, Predator, max_arch, use_fitness=True, fitness=fitness)
    print(color('%d\'', 'g') % iter, 'Archive: %d' % len(Archive))
    last_Prey = Prey.copy()
    last_Archive = Archive.copy()
    iter += 1

print(Archive)
Archive_fitness = np.array([fitness(p) for p in Archive])
print(Archive_fitness)
print(np.max(Archive, axis=0))


def draw_3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Archive_fitness[:, 0], Archive_fitness[:, 1], Archive_fitness[:, 2], c='r', marker='o')
    ax.set_xlabel('energy')
    ax.set_ylabel('times')
    ax.set_zlabel('quality')
    plt.show()


def draw_2d():
    plt.scatter(Archive_fitness[:, 0], Archive_fitness[:, 2], c='r', marker='o')
    plt.xlabel('energy')
    plt.ylabel('quality')
    plt.show()
    plt.scatter(Archive_fitness[:, 0], Archive_fitness[:, 1], c='r', marker='o')
    plt.xlabel('energy')
    plt.ylabel('times')
    plt.show()
    plt.scatter(Archive_fitness[:, 1], Archive_fitness[:, 2], c='r', marker='o')
    plt.xlabel('times')
    plt.ylabel('quality')
    plt.show()


end_time = time.time()
print('time: %f' % (end_time - start_time))
draw_3d()
draw_2d()
