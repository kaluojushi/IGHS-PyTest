import numpy as np
import pandas as pd

from print_lib import *

from scipy.optimize import minimize, Bounds


def main(Archive_fitness, E2T, E2Q, T2Q):
    print(color('-' * 15 + ' Sorting start ' + '-' * 15, 'y'))
    N = len(Archive_fitness)
    Mu1 = AHP(E2T, E2Q, T2Q)
    if Mu1 is not None:
        print('AHP weights: ' + color(Mu1, 'g'))
        Mu2 = G1(E2T, E2Q)
        print('G1 weights: ' + color(Mu2, 'g'))
        Mu3 = entropy_weight(Archive_fitness)
        print('entropy-weighting-method weights: ' + color(Mu3, 'g'))
        Mu4 = CRITIC(Archive_fitness)
        print('CRITIC weights: ' + color(Mu4, 'g'))
        x, W = SLSQP(Mu1, Mu2, Mu3, Mu4)
        print('SLSQP given coefficients: ' + color(x, 'c'))
        print('SLSQP weights: ' + color(W, 'c'))
        print(color('start sorting...', 'c'))
        ranks, scores = TOPSIS(Archive_fitness, W)
        print('TOPSIS ranks: ' + color(' '.join(str(x) for x in ranks[:8]) + ' ...(%d solutions total)' % N, 'g'))
        print(color('-' * 15 + ' Sorting end ' + '-' * 15, 'y'))
        return ranks, scores
    else:
        print(color('AHP uses inconsistent matrix!!!', 'r'))
        return None


def AHP(E2T, E2Q, T2Q):
    m = 3
    A = np.array([[1, E2T, E2Q], [1 / E2T, 1, T2Q], [1 / E2Q, 1 / T2Q, 1]])
    V, D = np.linalg.eig(A)
    tzz = np.max(V)  # The largest eigenvalue
    k = [i for i in range(m) if V[i] == tzz][0]  # The index of the largest eigenvalue
    tzx = D[:, k]  # The corresponding eigenvector
    CI = (tzz - m) / (m - 1)  # Consistency index
    RI = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59]
    CR = CI / RI[m - 1]  # Consistency ratio
    if CR < 0.1:
        return tzx / np.sum(tzx)
    else:
        return None


def G1(E2T, E2Q):
    m = 3
    order = np.zeros(3, dtype=int)
    if E2T >= 1 and E2Q >= 1:
        order[0] = 0
        T2Q = E2Q / E2T
        if T2Q >= 1:
            order[1:3] = [1, 2]
            k = np.array([E2T, np.round(T2Q)])
        else:
            order[1:3] = [2, 1]
            k = np.array([E2Q, np.round(1 / T2Q)])
    elif E2T >= 1 and E2Q < 1:
        order = np.array([2, 0, 1])
        k = np.array([np.round(1 / E2Q), E2T])
    elif E2T < 1 and E2Q >= 1:
        order = np.array([1, 0, 2])
        k = np.array([np.round(1 / E2T), E2Q])
    else:
        order[2] = 0
        T2Q = E2Q / E2T
        if T2Q >= 1:
            order[0:2] = [1, 2]
            k = np.array([np.round(T2Q), np.round(1 / E2Q)])
        else:
            order[0:2] = [2, 1]
            k = np.array([np.round(1 / T2Q), np.round(1 / E2T)])
    ratio = np.zeros(m - 1)
    for i in range(m - 1):
        ratio[i] = 9 / (10 - k[i])
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
    return weight


def entropy_weight(Archive_fitness):
    m, n = 3, len(Archive_fitness)
    Y = normalize_fitness(Archive_fitness)
    G = Y / np.sum(Y, axis=0)
    G_ = G * np.log(np.where(G == 0, 0.001, G))
    E = np.sum(G_, axis=0) / -np.log(n)
    w = (1 - E) / np.sum(1 - E)
    return w


def CRITIC(Archive_fitness):
    Y = normalize_fitness(Archive_fitness)
    sigma = np.std(Y, axis=0, ddof=1)
    R = np.corrcoef(Y, rowvar=False)
    F = np.sum(1 - R, axis=0)
    C = sigma * F
    w = C / np.sum(C)
    return w


def normalize_fitness(Archive_fitness):
    return (np.max(Archive_fitness, axis=0) - Archive_fitness) / (
                np.max(Archive_fitness, axis=0) - np.min(Archive_fitness, axis=0))


def SLSQP(Mu1, Mu2, Mu3, Mu4):
    m, p = 3, 4
    Mu1_ = Mu1.reshape((m, 1))
    Mu2_ = Mu2.reshape((m, 1))
    Mu3_ = Mu3.reshape((m, 1))
    Mu4_ = Mu4.reshape((m, 1))
    A = np.hstack((Mu1_, Mu2_, Mu3_, Mu4_))

    def obj(x):
        X = x.reshape((p, 1))
        w = A @ X
        S = np.sqrt(np.sum((w - Mu1_) ** 2)) + np.sqrt(np.sum((w - Mu2_) ** 2)) +\
            np.sqrt(np.sum((w - Mu3_) ** 2)) + np.sqrt(np.sum((w - Mu4_) ** 2))
        return S

    def cons(x):
        return x.sum() - 1

    x0 = np.ones(p) / p
    bnds = Bounds([0] * p, [1] * p)
    result = minimize(obj, x0, method='SLSQP', bounds=bnds, constraints={'type': 'eq', 'fun': cons})
    x = result.x
    X = x.reshape((p, 1))
    w = (A @ X).reshape((m,))
    return x, w


def TOPSIS(Archive_fitness, W):
    n = len(Archive_fitness)
    Y = normalize_fitness(Archive_fitness)
    Z = Y * W
    Z_plus = np.max(Z, axis=0)
    Z_minus = np.min(Z, axis=0)
    delta_Z_plus = np.abs(Z - Z_plus)
    delta_Z_minus = np.abs(Z - Z_minus)
    lambda_Z_plus = Z / Z_plus
    lambda_Z_minus = np.where(Z == 0, 1, Z_minus / np.where(Z == 0, 0.001, Z))
    gamma1_plus = 1 / np.exp(delta_Z_plus)
    gamma1_minus = 1 / np.exp(delta_Z_minus)
    gamma2_plus = 1 / np.exp(np.abs(1 - lambda_Z_plus))
    gamma2_minus = 1 / np.exp(np.abs(1 - lambda_Z_minus))
    gamma_plus = np.sum(gamma1_plus * gamma2_plus, axis=1) / n
    gamma_minus = np.sum(gamma1_minus * gamma2_minus, axis=1) / n
    scores = gamma_minus / (gamma_plus + gamma_minus)
    ranks = np.argsort(scores)[::-1]
    return ranks, scores[ranks]



if __name__ == '__main__':
    Archive_fitness = pd.read_csv('./output/exp_20230415-212818/Archive_20230415-212818_n=89.csv').values
    Archive_fitness = Archive_fitness[:, 5:8]
    E2T, E2Q, T2Q = 2, 1/3, 1/6
    main(Archive_fitness, E2T, E2Q, T2Q)
    # W = CRITIC(Archive_fitness)
    # print(W)
    # ranks, scores = TOPSIS(Archive_fitness, W)
    # Archive_sorted_fitness = Archive_fitness[ranks]
    # n = len(Archive_fitness)
    # info = np.hstack((Archive_sorted_fitness, scores.reshape((n, 1))))
    # df = pd.DataFrame(info, columns=['E', 'T', 'Q', 'score'])
    # print("CRITIC")
    # df.to_csv('./output/compare_20230417/CRITIC.csv', index=False, header=False)
