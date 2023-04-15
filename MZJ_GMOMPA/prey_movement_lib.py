import numpy as np
import scipy.special as sp


def update_prey(Prey, Predator, N, X_max, X_min, iter, max_iter, Hob, use_FADs=False):
    P = 0.5
    CF = np.power(1 - iter / max_iter, 2 * iter / max_iter)
    FADs = 0.2
    Prey_copy = Prey.copy()
    for i in range(N):
        if use_FADs:
            r = np.random.uniform(0, 1)
            if r < FADs:
                R = get_uniform_random()
                U = get_binary_random()
                Prey[i, :4] = Prey[i, :4] + CF * (X_min + R * (X_max - X_min)) * U
            else:
                r1, r2 = np.random.randint(0, N, 2)
                Prey[i, :4] = Prey[i, :4] + (FADs * (1 - r) + r) * (Prey_copy[r1, :4] - Prey_copy[r2, :4])
        else:
            if iter < max_iter / 3:
                RB = get_brownian_random()
                R = get_uniform_random()
                step_size = RB * (Predator[i, :4] - RB * Prey[i, :4])
                Prey[i, :4] = Prey[i, :4] + P * R * step_size
            elif iter < max_iter * 2 / 3:
                if i < N / 2:
                    RL = get_levy_random()
                    R = get_uniform_random()
                    step_size = RL * (Predator[i, :4] - RL * Prey[i, :4])
                    Prey[i, :4] = Prey[i, :4] + P * R * step_size
                else:
                    RB = get_brownian_random()
                    step_size = RB * (RB * Predator[i, :4] - Prey[i, :4])
                    Prey[i, :4] = Predator[i, :4] + P * CF * step_size
            else:
                RL = get_levy_random()
                step_size = RL * (RL * Predator[i, :4] - Prey[i, :4])
                Prey[i, :4] = Predator[i, :4] + P * CF * step_size
    Prey = rebound(Prey, N, X_max, X_min)
    Prey = attraction(Prey, N, X_max, X_min, Hob)
    return Prey


# Adjust the prey position to ensure that it is within the search range
# I call it the rebound algorithm
# 调整 Prey 的位置，确保其在搜索范围内
# 我称之为反弹算法
def rebound(Prey, N, X_max, X_min):
    span = X_max - X_min
    for i in range(N):
        for j in range(4):
            if span[j] == 0:
                Prey[i][j] = X_min[j]
            elif Prey[i][j] < X_min[j]:
                t = X_min[j] - Prey[i][j]
                _, tmp = divmod(t, 2 * span[j])
                Prey[i][j] = X_min[j] + tmp if tmp < span[j] else X_max[j] - tmp + span[j]
            elif Prey[i][j] > X_max[j]:
                t = Prey[i][j] - X_max[j]
                _, tmp = divmod(t, 2 * span[j])
                Prey[i][j] = X_max[j] - tmp if tmp < span[j] else X_min[j] + tmp - span[j]
    return Prey


# Adjust the prey position to ensure that it can meet the accuracy requirements
# I call it the attraction algorithm
# 调整 Prey 的位置，确保其可以满足精度要求
# 我称之为吸引力算法
def attraction(Prey, N, X_max, X_min, Hob):
    Hob_filter = Hob[
        np.where((X_min[0] <= Hob[:, 0]) & (Hob[:, 0] <= X_max[0]) & (X_min[1] <= Hob[:, 1]) & (Hob[:, 1] <= X_max[1]))]
    Hob_filter_max = np.max(Hob_filter, axis=0)
    Hob_filter_min = np.min(Hob_filter, axis=0)
    Hob_filter_rescale = np.where(Hob_filter_max - Hob_filter_min == 0, 0.5,
                                  (Hob_filter - Hob_filter_min) / np.where(Hob_filter_max - Hob_filter_min == 0, 1e-10,
                                                                           Hob_filter_max - Hob_filter_min))
    for i in range(N):
        hob = Prey[i, 0:2]
        hob_rescale = np.where(Hob_filter_max - Hob_filter_min == 0, 0.5,
                               (hob - Hob_filter_min) / np.where(Hob_filter_max - Hob_filter_min == 0, 1e-10,
                                                                 Hob_filter_max - Hob_filter_min))
        distances = np.sqrt(np.sum(np.square(Hob_filter_rescale - hob_rescale), axis=1))
        index = np.argmin(distances)
        Prey[i, 0:2] = Hob_filter[index]
        Prey[i, 2] = np.round(Prey[i, 2])
        Prey[i, 3] = np.round(Prey[i, 3] * 100) / 100
    return Prey


# Get a random Brownian vector
# Each element is normally distributed with a mean of 0 and a standard deviation of 1
# 获得一个随机的 Brownian 向量，每个元素正态分布，均值为 0，标准差为 1
def get_brownian_random():
    return np.random.normal(0, 1, 4)


# Get a random Levy vector
# 获得一个随机的 Levy 向量
def get_levy_random():
    alpha = 1.5
    sigma_x = np.power(
        (sp.gamma(1 + alpha) * np.sin(np.pi * alpha / 2)) / (
                sp.gamma((1 + alpha) / 2) * alpha * np.power(2, (alpha - 1) / 2)),
        1 / alpha)
    x = np.random.normal(0, sigma_x, 4)
    y = np.random.normal(0, 1, 4)
    BL = 0.05 / np.power(np.linalg.norm(y), 1 / alpha) * x
    return BL


# Get a random uniform vector
# Each element is uniformly distributed in the range [0, 1]
# 获得一个随机的均匀向量，每个元素在 [0, 1] 范围内均匀分布
def get_uniform_random():
    return np.random.uniform(0, 1, 4)


# Get a random binary vector
# Each element has a 20% probability of being 0, and an 80% probability of being 1
# 获得一个随机的二进制向量，每个元素有 20% 的概率为 0，80% 的概率为 1
def get_binary_random():
    return np.array([0 if np.random.uniform(0, 1) < 0.2 else 1 for _ in range(4)])


if __name__ == '__main__':
    X_max = np.array([2, 89, 754, 2.6], dtype=np.float64) + 1e-5
    X_min = np.array([2, 76, 646, 2], dtype=np.float64)
    Hob = np.array([[1, 70.25], [1, 75.5], [1, 78], [1, 80.5], [1, 83.5], [1, 85.5], [1, 89], [1, 90.2],
                    [2, 70.25], [2, 75.5], [2, 78], [2, 80.5], [2, 83.5], [2, 85.5], [2, 89], [2, 90.2],
                    [3, 70.25], [3, 75.5], [3, 78], [3, 80.5], [3, 83.5], [3, 85.5], [3, 89], [3, 90.2]],
                   dtype=np.float64)
    N = 10
    Prey = np.array([[np.random.uniform(X_min[i], X_max[i]) for i in range(4)] for _ in range(N)], dtype=np.float64)
    attraction(Prey, N, X_max, X_min, Hob)
