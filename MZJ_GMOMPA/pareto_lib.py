import numpy as np


# Determine if solution u dominates another solution v
# If u dominates v, that is, each objective function value of u is not worse than v
# 确定解 u 是否支配另一个解 v
# 如果 u 支配 v，即 u 的每个目标函数值都不比 v 差
def dominates(u, v):
    uf, vf = u[4:], v[4:]
    return np.all(uf <= vf) and np.any(uf < vf)


# Determine if solution u is non-dominated in a solution set
# If u is non-dominated, that is, u is not dominated by any solution in the solution set
# 确定解 u 是否在解集中是非支配的
# 如果 u 是非支配的，即 u 不被解集中的任何解支配
def is_non_dominated(up, solution_set):
    for vp in solution_set:
        if dominates(vp, up):
            return False
    return True


# Get the non-dominated solution set from a solution set
# The non dominated solution set is a subset of the solution set
# Including all non-dominated solutions of the solution set
# Each of which is not dominated by each other
# 从解集中获取非支配解集
# 非支配解集是解集的一个子集，包括解集中所有非支配解，每个解都不被其他解支配
def get_non_dominated_set(solution_set):
    return np.array([up for up in solution_set if is_non_dominated(up, solution_set)])


# Test
if __name__ == '__main__':
    pass
