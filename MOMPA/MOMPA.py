import random

upper_limit = [3, 80, 410, 1.6]
lower_limit = [2, 75, 375, 1.5]
u = 0.9
for i in [1, 2]:
    y, x = upper_limit[i], lower_limit[i]
    lower_limit[i] = u * x
    upper_limit[i] = y + (1 - u) * x

N = 100





def generate_population(N):
    population = []
    for i in range(N):
        population.append([random.uniform(lower_limit[0], upper_limit[0]),
                           random.uniform(lower_limit[1], upper_limit[1]),
                           random.uniform(lower_limit[2], upper_limit[2]),
                           random.uniform(lower_limit[3], upper_limit[3])])
    return population
