import numpy as np
from matplotlib import pyplot as plt

from print_lib import *
import predator_lib as pl
import prey_movement_lib as pml


def main(N, max_arch, max_iter, ul, ll, expand_rate, Hob, fitness, constraints):
    print(color('-' * 15 + ' Optimization start ' + '-' * 15, 'y'))
    ul, ll = expand_search_space(ul, ll, expand_rate)
    iter = -1
    Prey, Predator, Archive, last_Prey, last_Archive = None, None, None, None, None
    print(color('start iteration...', 'c'))
    min_fitness = np.zeros((max_iter + 1, 3))
    while iter < max_iter:
        print(color('*' * 5 + ' iter: %d ' % iter + '*' * 5, 'p'))
        # If it is the initial iteration, build the prey population
        # Otherwise, prey population moves
        # 如果是最初迭代，构建 Prey 种群；否则，Prey 种群移动
        if iter == -1:
            Prey = build_prey(N, ul, ll)
            Prey = pml.attraction(Prey, N, X_max=ul, X_min=ll, Hob=Hob)
            Prey = update_population_fitness(Prey, fitness)
            Apop = Prey.copy()
            print(color('%d\'', 'p') % iter, 'Prey built successfully!')
        else:
            Prey = pml.update_prey(Prey, Predator, N, X_max=ul, X_min=ll, iter=iter, max_iter=max_iter,
                                   Hob=Hob)
            Prey = pml.update_prey(Prey, Predator, N, X_max=ul, X_min=ll, iter=iter, max_iter=max_iter,
                                   Hob=Hob, use_FADs=True)
            Prey = update_population_fitness(Prey, fitness)
            Apop = pl.construct_Apop(Prey, last_Prey, last_Archive)
            print(color('%d\'', 'p') % iter, 'Prey moved successfully!')
        # After getting Apop, sort it, find Predator population, and update Archive
        # 得到 Apop 后，对其进行排序，找到 Predator 种群，更新 Archive
        sorted_layers = pl.get_sorted_front_layers(Apop)
        Apop = pl.get_sorted_new_population(Apop, sorted_layers)
        print(color('%d\'', 'p') % iter,
              'Apop with ' + color('%d' % len(Apop), 'g') + ' solutions built ans sorted successfully!')
        Predator = pl.construct_predator_population(Apop[sorted_layers[0]], N, constraints)
        print(color('%d\'', 'p') % iter, 'Predator built successfully!')
        if iter == -1:
            print(color('  start update Archive...', 'c'))
            print('  Archive start length: ' + color('%d' % 0, 'g'))
            Archive = Apop[sorted_layers[0]]
            print('  Archive end length: ' + color('%d' % len(Archive), 'g'))
        else:
            Archive = pl.update_archive_population(Archive, Predator, max_arch)
        print(color('%d\'', 'p') % iter,
              'Archive with ' + color('%d' % len(Archive), 'g') + ' solutions updated successfully!')
        for j in range(3):
            min_fitness[iter + 1, j] = np.min(Archive[:, 4 + j])
        last_Prey = Prey.copy()
        last_Archive = Archive.copy()
        iter += 1
    print(color('*' * 5 + ' iter: %d ' % max_iter + '*' * 5, 'p'))
    print(color('iteration finished.', 'c'))
    print('Archive with ' + color('%d' % len(Archive), 'g') + ' solutions obtained successfully!')
    print(color('-' * 15 + ' Optimization end ' + '-' * 15, 'y'))
    return Archive, min_fitness


def expand_search_space(ul, ll, expand_rate):
    for i in [1, 2]:
        y, x = ul[i], ll[i]
        ll[i] = expand_rate * x
        ul[i] = y + (1 - expand_rate) * x
    # Make sure that the search space meets the accuracy requirements
    # 确保搜索空间满足精度要求
    for i in [0, 2]:
        ll[i] = np.ceil(ll[i])
        ul[i] = np.floor(ul[i])
    for i in [1, 3]:
        ll[i] = np.ceil(ll[i] * 100) / 100
        ul[i] = np.floor(ul[i] * 100) / 100
    # Add a small value to the upper limit to make sure that the upper limit is included in the search space
    # 为上限添加一个小值，以确保上限包含在搜索空间中
    ul += 1e-5
    print('search space expanded(rate=%f) to: ' % expand_rate)
    print('upper limit: ', color(ul, 'g'))
    print('lower limit: ', color(ll, 'g'))
    return ul, ll


def build_prey(N, ul, ll):
    Prey = np.zeros((N, 7))
    for i in range(N):
        for j in range(4):
            Prey[i, j] = np.random.uniform(ll[j], ul[j])
    return Prey


def update_population_fitness(population, fitness):
    for i in range(len(population)):
        population[i, 4:] = fitness(population[i, :4])
    return population


def draw_energy(Archive, iter):
    energy = Archive[:, 4][np.argsort(Archive[:, 4])]
    plt.scatter(range(len(energy)), energy, marker='.')
    plt.title('iter: %d' % iter)
    plt.show()
