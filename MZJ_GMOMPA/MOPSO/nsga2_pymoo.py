import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from objectives import get_objective, energy_and_time


def solve_problem(problem):
    algorithm = NSGA2(pop_size=100)
    gens = 150 if problem == "zdt2" else 300

    # API call
    res = minimize(problem, algorithm, ('n_gen', gens), seed=1, verbose=True)
    print("Running time of NSGA2 :", res.exec_time, "s")
    print("res.X:", res.X)
    print("res.F:", res.F)
    print("res:", res)
    np.savetxt("./coords417/nsga2s_in.txt", res.X)
    df_in = pd.DataFrame(res.X)
    df_in.to_csv("./coords417/nsga2s_in.csv", index=False)
    np.savetxt("./coords417/nsga2s_fit.txt", res.F)
    df_fit = pd.DataFrame(res.F)
    df_fit.to_csv("./coords417/nsga2s_fit.csv", index=False)
    #
    # plt.title('NSGA2_Front')
    # plt.xlabel('fitness_y1')
    # plt.ylabel('fitness_y2')
    # plt.scatter(res.F[:, 0], res.F[:, 1], s=30, c='red', marker=".", alpha=1.0)
    #
    # plt.savefig('./imgs/NSGA2_Front.png')
    # plt.close()


def driver(funct):
    if funct == 1:
        problem = get_problem("zdt2")
    elif funct == 2:
        problem = get_problem("tnk")
    elif funct == 3:
        problem = get_problem("osy")
    elif funct == 4:
        problem = get_problem("bnh")
    else:
        # IGHS
        problem = MyProblem()

    solve_problem(problem)


def my_func(in_):
    return get_objective("value", 5, 3, in_)


def my_cons(in_):
    cons = np.zeros((len(in_), 5))
    for i in range(len(in_)):
        z0, da0, n0, f = in_[i]
        vc = np.pi * da0 * n0 / 1000
        u = np.array([2.5, 45, 0.349, 0.297, 128.63, 45, 8.45])
        _, _, Pc1, Pc2 = energy_and_time(in_[i], u)
        Fc1, Fc2 = max(Pc1) * 60 / vc, max(Pc2) * 60 / vc
        Fc = max(Fc1, Fc2)
        Fc_max = 1e5
        cons[i, 0] = Fc - Fc_max
        hob = np.array([76.5, 78, 80.5, 83, 86.5, 88.5])
        temp = np.any(np.abs(hob - da0) < 1e-2)
        cons[i, 2] = -1 if temp else 1
    temp = np.multiply(in_[:, 3], in_[:, 3])
    temp = np.multiply(temp, 0.0312)
    temp = np.divide(temp, 0.6)
    cons[:, 1] = temp - 1.6
    cons[:, 3] = np.abs(in_[:, 2] - np.round(in_[:, 2])) - 1e-2
    cons[:, 4] = np.abs(in_[:, 3] - np.round(in_[:, 3], 2)) - 1e-3
    return cons


class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=4, n_obj=3, n_ieq_constr=5, xl=np.array([2, 74.1, 609, 1.8]), xu=np.array([2, 90.4, 732, 2.6]) + 1e-5)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = my_func(x)
        out["G"] = my_cons(x)


if __name__ == '__main__':
    driver(5)
