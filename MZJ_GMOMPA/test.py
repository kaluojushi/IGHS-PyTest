import numpy as np

in_ = np.array([[1, 2, 3.000012, 4.2],
                [1, 2, 3.1, 4.2]])
cons = np.zeros((len(in_), 5))
cons[:, 3] = np.abs(in_[:, 2] - np.round(in_[:, 2])) - 1e-2
cons[:, 4] = np.abs(in_[:, 3] - np.round(in_[:, 3], 2)) - 1e-5

print(cons)
