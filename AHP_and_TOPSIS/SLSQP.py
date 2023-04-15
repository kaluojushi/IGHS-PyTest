import numpy as np
from scipy.optimize import minimize, Bounds

def objective(x):
    W1 = np.array([[0.426, 0.212, 0.097, 0.168, 0.097]]).T
    W2 = np.array([[0.2044, 0.2093, 0.1767, 0.2036, 0.2060]]).T
    # W3 = np.array([[0.3364, 0.3922, 0.0737, 0.0858, 0.1119]]).T
    W3 = np.array([[0.3, 0.2, 0.1, 0.1, 0.3]]).T
    A = np.hstack((W1, W2, W3))
    X = np.reshape(x, (3, 1))
    w = A @ X
    S = np.sqrt(np.sum((w - W1) ** 2)) + np.sqrt(np.sum((w - W2) ** 2)) + np.sqrt(np.sum((w - W3) ** 2))
    # S = np.sum((w - W1) ** 2) + np.sum((w - W2) ** 2) + np.sum((w - W3) ** 2)
    # S = np.sum((w - W1) ** 2) * np.sum((w - W2) ** 2) * np.sum((w - W3) ** 2)
    return S

def constraint(x):
    return x.sum() - 1  # 约束条件

# 初始值
x0 = np.array([0.5, 0.3, 0.2])

# 定义变量的边界，即每个xi的取值范围
bounds = Bounds([0, 0, 0], [1, 1, 1])

# 求解
result = minimize(objective, x0, method='SLSQP', constraints={'fun': constraint, 'type': 'eq'}, bounds=bounds)

# 打印结果
print(result)
print('x: ', result.x)
