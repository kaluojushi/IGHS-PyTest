import numpy as np


# Determine if solution u epsilon-dominates another solution v
# If u epsilon-dominates v, that is, each objective function value of u is not worse than v by more than epsilon
# e: a vector of epsilon values, one for each objective function
# 确定解 u 是否以 epsilon 支配另一个解 v
# 如果 u 以 epsilon 支配 v，即 u 的每个目标函数值都不比 v 差超过 epsilon
# e: 每个目标函数的 epsilon 值的向量
def epsilon_dominates(u, v, e, use_fitness=False, fitness=None):
    uf, vf = (fitness(u), fitness(v)) if use_fitness else (u, v)
    return np.all(uf <= vf + e) and np.any(uf < vf + e)

