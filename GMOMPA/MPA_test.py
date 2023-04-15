import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

d = 2
dtype = np.dtype([('p', np.float64, d), ('f', np.float64)])
lower_limit = 0
upper_limit = 7


def fitness(a):
    return np.sum(np.square(a))


def generate_population(N):
    population = np.zeros(N, dtype=dtype)
    for i in range(N):
        population[i]['p'] = np.random.uniform(lower_limit, upper_limit, d)
        population[i]['f'] = fitness(population[i]['p'])
    return population


def update_elite(population=None, Prey=None, Elite=None, N=0, is_start=False):
    if is_start:
        sorted_indices = np.argsort(population['f'])[::-1]
        elite_index = sorted_indices[0]
        elite = population[elite_index]
        Elite = np.tile(elite['p'], (N, 1))
        return Elite
    else:
        fitness_list = np.array([fitness(p) for p in Prey])
        sorted_indices = np.argsort(fitness_list)[::-1]
        elite_index = sorted_indices[0]
        elite = Prey[elite_index]
        if fitness(elite) > fitness(Elite[0]):
            Elite = np.tile(elite, (N, 1))
        return Elite


def build_prey(population):
    Prey = population['p'].copy()
    return Prey


def update_prey(Elite, Prey, N, iter, max_iter, mode):
    P = 0.5
    CF = np.power(1 - iter / max_iter, 2 * iter / max_iter)
    FADs = 0.2
    Prey_copy = Prey.copy()
    for i in range(N):
        if mode == 1:
            RB = get_brownian_random()
            R = get_uniform_random()
            step_size = RB * (Elite[i] - RB * Prey[i])
            Prey[i] = Prey[i] + P * R * step_size
        elif mode == 2:
            if i < N / 2:
                RL = get_levy_random()
                R = get_uniform_random()
                step_size = RL * (Elite[i] - RL * Prey[i])
                Prey[i] = Prey[i] + P * R * step_size
            else:
                RB = get_brownian_random()
                step_size = RB * (RB * Elite[i] - Prey[i])
                Prey[i] = Elite[i] + P * CF * step_size
        elif mode == 3:
            RL = get_levy_random()
            step_size = RL * (RL * Elite[i] - Prey[i])
            Prey[i] = Elite[i] + P * CF * step_size
        elif mode == 4:
            r = np.random.uniform(0, 1)
            if r < FADs:
                R = get_uniform_random()
                U = get_binary_random()
                X_max = np.tile(10, d)
                X_min = np.tile(0, d)
                Prey[i] = Prey[i] + CF * (X_min + R * (X_max - X_min)) * U
            else:
                r1, r2 = np.random.randint(0, N, 2)
                Prey[i] = Prey[i] + (FADs * (1 - r) + r) * (Prey_copy[r1] - Prey_copy[r2])
    prey_out_of_range_check(Prey, N)


def get_brownian_random():
    return np.random.normal(0, 1, d)


def get_levy_random():
    alpha = 1.5
    sigma_x = np.power(
        (gamma(1 + alpha) * np.sin(np.pi * alpha / 2)) / (
                gamma((1 + alpha) / 2) * alpha * np.power(2, (alpha - 1) / 2)),
        1 / alpha)
    x = np.random.normal(0, sigma_x, d)
    y = np.random.normal(0, 1, d)
    BL = 0.05 / np.power(np.linalg.norm(y), 1 / alpha) * x
    return BL


def get_uniform_random():
    return np.random.uniform(0, 1, d)


def get_binary_random():
    return np.array([0 if np.random.uniform(0, 1) < 0.2 else 1 for _ in range(d)])


def prey_out_of_range_check(Prey, N):
    span = upper_limit - lower_limit
    for i in range(N):
        for j in range(d):
            if Prey[i][j] < lower_limit:
                Prey[i][j] = 2 * lower_limit - Prey[i][j]
            if Prey[i][j] > upper_limit:
                t = Prey[i][j] - upper_limit
                _, tmp = divmod(t, 2 * span)
                Prey[i][j] = upper_limit - tmp if tmp < span else lower_limit + tmp - span


N = 20
population = generate_population(N)
# print(population)
Elite = update_elite(population=population, N=N, is_start=True)
elites = np.array([Elite[0]])
print('Initial Elite:', Elite[0])
print('Initial Elite Fitness:', fitness(Elite[0]))
Prey = build_prey(population)
preys = np.array([Prey])

iter = 0
max_iter = 100
while iter < max_iter:
    if iter < max_iter / 3:
        mode = 1
    elif iter < max_iter * 2 / 3:
        mode = 2
    else:
        mode = 3
    update_prey(Elite, Prey, N, iter, max_iter, mode=mode)
    # print(iter, mode, Prey)
    Elite = update_elite(Prey=Prey, Elite=Elite, N=N)
    update_prey(Elite, Prey, N, iter, max_iter, mode=4)
    preys = np.append(preys, [Prey], axis=0)
    Elite = update_elite(Prey=Prey, Elite=Elite, N=N)
    elites = np.append(elites, [Elite[0]], axis=0)
    iter += 1

print('Final Elite:', Elite[0])
print('Final Elite Fitness:', fitness(Elite[0]))
print('elites:', elites)


def draw_elites():
    X = elites[:, 0]
    Y = elites[:, 1]
    Z = [fitness(elite) for elite in elites]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(X, Y, Z, 'gray')
    ax.scatter3D(X, Y, Z, c=Z, cmap='viridis', linewidth=0.5)
    fig.colorbar(ax.scatter3D(X, Y, Z, c=Z, cmap='viridis', linewidth=0.5), shrink=0.5, aspect=5)
    plt.show()


def draw_preys():
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    for i in range(N):
        X = preys[:, i, 0]
        Y = preys[:, i, 1]
        # Z = [fitness(prey) for prey in preys[:, i]]
        # ax.plot3D(X, Y, Z, 'gray')
        # ax.scatter3D(X, Y, Z, c=Z, cmap='viridis', linewidth=0.5)
        plt.scatter(X[0], Y[0], c='b')
        plt.plot(X, Y, ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w'][i % 8] + '-', linewidth=0.5)
        plt.scatter(X[-1], Y[-1], c='g')
        if i == 0: break
    plt.show()

draw_elites()
draw_preys()
