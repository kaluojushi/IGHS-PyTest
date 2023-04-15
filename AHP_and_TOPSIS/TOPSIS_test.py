import numpy as np


def normalize_positively(X):
    Y = X.copy()
    for j in range(M):
        if objective_types[j] == 'min':
            Y[:, j] = (np.max(Y[:, j]) - Y[:, j]) / (np.max(Y[:, j]) - np.min(Y[:, j]))
        else:
            Y[:, j] = (Y[:, j] - np.min(Y[:, j])) / (np.max(Y[:, j]) - np.min(Y[:, j]))
    return Y


def normalize_X(X):
    X_ = X.copy()
    for j in range(M):
        if objective_types[j] == 'min':
            X_[:, j] = np.max(X_[:, j]) - X_[:, j]
    return X_ / np.sqrt(np.sum(X_ ** 2, axis=0))


N = 10
M = 3
objective_types = ['min', 'min', 'max']

X = np.array([[7.5, 210, 4.1],
              [7.9, 250, 4.3],
              [8.1, 190, 3.9],
              [8.1, 205, 4.2],
              [7.3, 230, 3.8],
              [6.9, 240, 3.6],
              [7.2, 220, 4.0],
              [8.5, 190, 4.5],
              [6.5, 220, 3.7],
              [7.2, 200, 3.9]])
Y = normalize_positively(X)
T = np.sqrt(np.sum((Y - np.mean(Y, axis=0)) ** 2, axis=0) / (N - 1)) / np.mean(Y, axis=0)
weight = T / np.sum(T)
print(weight)
X_ = normalize_X(X)
print(X_)
Z = X_ * weight
print(Z)
Z_plus = np.max(Z, axis=0)
Z_minus = np.min(Z, axis=0)
print(Z_plus, Z_minus)
D_plus = np.sqrt(np.sum((Z - Z_plus) ** 2, axis=1))
D_minus = np.sqrt(np.sum((Z - Z_minus) ** 2, axis=1))
print(D_plus, D_minus)
C = D_minus / (D_plus + D_minus)
print(C)
C = C / np.sum(C)
print(C)
ranks = np.argsort(C)[::-1]
print(ranks)
print(X[ranks])

