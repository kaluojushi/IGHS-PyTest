from Archive import *


def objective_positively(X):
    X_pos = X.copy()
    for j in range(M):
        if objective_types[j] == 'min':
            X_pos[:, j] = np.max(X_pos[:, j]) - X_pos[:, j]
    return X_pos


def normalize_X(X):
    return X / np.sqrt(np.sum(X ** 2, axis=0))


N = len(Archive_params)
M = 3
objective_types = ['min', 'min', 'min']
# N = 4
# M = 2
# objective_types = ['max', 'min']

X = Archive_fitness
# X = np.array([[99, 0], [60, 10], [89, 2], [74, 3]], dtype=float)
X_pos = objective_positively(X)
Z = normalize_X(X_pos)
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
print(Archive_params[ranks])
