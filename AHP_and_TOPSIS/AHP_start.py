import numpy as np
from Archive import *


def calculate_tzz_and_tzx(mat):
    n = len(mat)
    V, D = np.linalg.eig(mat)

    tzz = np.max(V)  # The largest eigenvalue
    print('tzz:', tzz)
    k = [i for i in range(n) if V[i] == tzz][0]  # The index of the largest eigenvalue
    tzx = D[:, k]  # The corresponding eigenvector
    print('tzx:', tzx)
    return tzz, tzx


def consistency_check(mat):
    n = len(mat)
    tzz, tzx = calculate_tzz_and_tzx(mat)
    CI = (tzz - n) / (n - 1)  # Consistency index
    RI = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59]
    CR = CI / RI[n - 1]  # Consistency ratio
    return CR < 0.1


def normalize_tzx(tzx):
    return tzx / np.sum(tzx)


# Give the importance of each objective value relative to the others
# X2Y represents the importance of X relative to Y
# If X2Y > 1, X is more important than Y
E2Q, E2T, Q2T = 1, 1, 1

N = len(Archive_params)
M = 3
A = np.array([[1, E2Q, E2T], [1 / E2Q, 1, Q2T], [1 / E2T, 1 / Q2T, 1]])
BE = np.array([[Archive_fitness[:, 0][i] / Archive_fitness[:, 0][j] for j in range(N)] for i in range(N)])
BT = np.array([[Archive_fitness[:, 1][i] / Archive_fitness[:, 1][j] for j in range(N)] for i in range(N)])
BQ = np.array([[Archive_fitness[:, 2][i] / Archive_fitness[:, 2][j] for j in range(N)] for i in range(N)])

if consistency_check(A):
    _, tzx_A = calculate_tzz_and_tzx(A)
    _, tzx_BE = calculate_tzz_and_tzx(BE)
    _, tzx_BT = calculate_tzz_and_tzx(BT)
    _, tzx_BQ = calculate_tzz_and_tzx(BQ)
    weight = normalize_tzx(tzx_A)
    np.reshape(weight, (M, 1))
    priorities = np.vstack((normalize_tzx(tzx_BE), normalize_tzx(tzx_BT), normalize_tzx(tzx_BQ))).T
    scores = priorities @ weight
    ranks = np.argsort(scores)[::-1]
    print('scores:', scores)
    print('ranks:', ranks)
    print('Archive_params:', Archive_params[ranks])
    print('Archive_fitness:', Archive_fitness[ranks])
else:
    print('Inconsistent matrix A')
