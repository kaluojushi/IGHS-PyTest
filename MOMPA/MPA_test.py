import numpy as np
from scipy.special import gamma

d = 2
dtype = np.dtype([('p', np.float64, d), ('f', np.float64)])


def fitness(a):
    return np.sum(np.square(a))


def generate_population(N):
    population = np.zeros(N, dtype=dtype)
    for i in range(N):
        population[i]['p'] = np.random.uniform(0, 10, d)
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
    for i in range(N):
        for j in range(d):
            if Prey[i][j] < 0:
                Prey[i][j] = -Prey[i][j]
            if Prey[i][j] > 10:
                _, tmp = divmod(Prey[i][j], 20)
                Prey[i][j] = tmp if tmp < 10 else 20 - tmp





N = 20
population = generate_population(N)
print(population)
Elite = update_elite(population=population, N=N, is_start=True)
print(Elite[0])
Prey = build_prey(population)

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
    print(iter, mode, Prey)
    Elite = update_elite(Prey=Prey, Elite=Elite, N=N)
    update_prey(Elite, Prey, N, iter, max_iter, mode=4)
    Elite = update_elite(Prey=Prey, Elite=Elite, N=N)
    iter += 1

print(Elite[0])
