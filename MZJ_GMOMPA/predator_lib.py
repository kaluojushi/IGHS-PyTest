import numpy as np
from pareto_lib import *
from epsilon_lib import *
from print_lib import *


# Construct Apop population
# Apop population is the union set of Prey, last_Prey and last_Archive
# 构造 Apop 种群
# Apop 种群是 Prey、last_Prey 和 last_Archive 的并集
def construct_Apop(*populations):
    # for p in populations:
    #     p[:, 4:] = np.zeros(3)
    # set(tuple(p)) is used to remove the duplicate solutions
    # list(set(tuple(p))) is used to avoid the error of np.vstack() because its input should be a sequence
    # set(tuple(p)) 用于去除重复的解
    # list(set(tuple(p))) 用于避免 np.vstack() 报错，因为它的输入应该是一个序列
    return np.vstack(list(set(tuple(p) for p in np.vstack(populations))))


# Fast non-dominated sorting of a population
# Output a sorted population based on multiple Pareto front layers
# The solutions of current front layer is only dominated by the solutions of the previous front layer
# 对一个种群进行快速非支配排序（FNS）
# 输出的是多个 Pareto 前沿层
# 当前前沿层的解只被前一层的解所支配
def fast_non_dominated_sort(population):
    N = len(population)
    # Step 1: Calculate Si and ni for each solution i
    # Si: the set of solutions that solution i dominates
    # ni: the number of solutions that dominate solution i
    # 步骤 1：计算每个解 i 的 Si 和 ni
    # Si：解 i 支配的解集；ni：支配解 i 的解的数量
    sets_of_ind_dominates = [[] for _ in population]
    number_of_dominates_ind = np.zeros(N, dtype=int)
    for i in range(N):
        x = population[i]
        set_of_x_dominates = sets_of_ind_dominates[i]
        for j in range(N):
            y = population[j]
            if dominates(x, y):
                set_of_x_dominates.append(j)
            elif dominates(y, x):
                number_of_dominates_ind[i] += 1
    # print('sets_of_ind_dominates:', sets_of_ind_dominates)
    # print('number_of_dominates_ind:', number_of_dominates_ind)
    # Step 2: Calculate each front layer
    # The solutions of current front layer is only dominated by the solutions of the previous front layer
    # 步骤 2：计算每个前沿层
    # 当前前沿层的解只被前一层的解所支配
    front_layers = []
    first_front_layer = np.where(number_of_dominates_ind == 0)[0]
    current_front_layer = first_front_layer.copy()
    while current_front_layer.size > 0:
        front_layers.append(current_front_layer)
        next_front_layer = []
        for i in current_front_layer:
            for j in sets_of_ind_dominates[i]:
                number_of_dominates_ind[j] -= 1
                if number_of_dominates_ind[j] == 0:  # which means no solution dominates j
                    next_front_layer.append(j)
        current_front_layer = np.array(next_front_layer)
    # Step 3: Output Pareto front layers
    # 步骤 3：输出 Pareto 前沿层
    return front_layers


# Sort the front layer using the crowding distance of each solution in this front layer
# The crowding distance of a solution is the sum of the distances between the solution and its two neighbors
# Output the sorted front layer, where each element is the index of the solution in the population
# 使用前沿层中每个解的拥挤距离对前沿层进行排序
# 解的拥挤距离是该解与其两个邻居之间的距离之和
# 输出的是排序后的前沿层，其中每个元素是该解在种群中的索引
def sort_layer_using_crowding_distances(population, front_layer):
    front_fitness = np.array([x[4:] for x in population[front_layer]])
    layer_size = len(front_layer)
    # If the size of the front layer is less than or equal to 2, no need to sort
    # 如果前沿层的大小小于等于 2，则不需要排序
    if layer_size <= 2:
        return front_layer
    M = len(front_fitness[0])  # number of objectives
    # distances[i] means the crowding distance of the solution front_layer[i]
    # front_layer[i] means the index of the solution in the population
    # distances[i] 表示个体 front_layer[i] 的拥挤距离
    # front_layer[i] 表示该解在种群中的索引
    distances = np.zeros(layer_size, dtype=float)
    for m in range(M):  # for each objective
        # Find the max and min value of the objective m and set the distance of these solutions to infinity
        # 找到目标 m 的最大值和最小值，并将这些解的距离设置为无穷大
        max_m = np.max(front_fitness[:, m])
        min_m = np.min(front_fitness[:, m])
        distances[np.where(front_fitness[:, m] == max_m)] = np.inf
        distances[np.where(front_fitness[:, m] == min_m)] = np.inf
        # Sort the solutions of the front layer by the objective m
        # The origin order of the front layer should not be changed
        # front_sorted_indices: the index of the front layer sorted by the objective m
        # 将前沿层的解按目标 m 排序
        # 前沿层的原始顺序不应该改变
        # front_sorted_indices：按目标 m 排序的前沿层的索引
        front_sorted_indices = np.argsort(front_fitness[:, m])
        # Update the distance of each solution
        # 更新每个解的距离
        if max_m != min_m:
            for i in range(1, layer_size - 1):
                cur, pre, nxt = front_sorted_indices[[i, i - 1, i + 1]]
                distances[cur] += (front_fitness[nxt, m] - front_fitness[pre, m]) / (max_m - min_m)
    # Sort the front layer by the crowding distance in descending order
    # The larger the crowding distance, the higher the priority given to the corresponding solution
    # 将前沿层按拥挤距离降序排序
    # 拥挤距离越大，对应解的优先级越高
    return front_layer[np.argsort(-distances)]


# Get sorted front layers
# 得到排序后的前沿层
def get_sorted_front_layers(population, front_layers=None):
    if front_layers is None:
        front_layers = fast_non_dominated_sort(population)
    # print('front_layers:', front_layers)
    sorted_layers = [sort_layer_using_crowding_distances(population, front_layer) for front_layer
                     in front_layers]
    # print('sorted_layers:', sorted_layers)
    return sorted_layers


# Get new population by selecting the solutions in the front layers
# The solutions in each front layer are sorted by the crowding distance
# 通过选择前沿层中的解来获得新的种群
# 每个前沿层中的解按拥挤距离排序
def get_sorted_new_population(population, sorted_layers=None):
    if sorted_layers is None:
        sorted_layers = get_sorted_front_layers(population)
    sorted_order = np.concatenate(sorted_layers)
    # print('sorted_order:', sorted_order)
    return population[sorted_order]


# Construct Predator population
# which is the first N solutions in the sorted population
# 构造 Predator 种群
# 它是排序后种群中的前 N 个解
def construct_predator_population(first_population, N, constraints):
    # If a solution violates the constraints, it will be filtered out
    # 如果一个解违反了约束条件，它将被过滤掉
    mask = np.apply_along_axis(constraints, 1, first_population[:, :4])
    filtered_first_population = first_population[np.all(mask, axis=1)]
    first_N = len(filtered_first_population)
    if first_N >= N:
        return filtered_first_population[:N]
    else:
        return filtered_first_population[np.append(np.arange(first_N), np.random.randint(0, first_N, N - first_N))]


# Update the Archive population
# 更新 Archive 种群
def update_archive_population(Archive, Predator, max_arch):
    print(color('  start update Archive...', 'c'))
    new_Archive = Archive.copy()
    print('  Archive start length: ' + color('%d' % len(new_Archive), 'g'))
    for i in range(len(Predator)):
        s = Predator[i]
        # Step 1: Calculate S, Se and n for current solution s
        # S: the set of solutions that solution s dominates
        # n: the number of solutions that dominate solution s
        # ne: the number of solutions that epsilon-dominate solution s
        # 步骤 1：计算当前解 s 的 S 和 n
        # S：解 s 支配的解集；n：支配解 s 的解的数量；ne：以 epsilon 支配解 s 的解的数量
        sets_of_s_dominates = []
        number_of_dominates_s = 0
        number_of_epsilon_dominates_s = 0
        Archive_fitness = Archive[:, 4:]
        max_fitness = np.array([np.max(Archive_fitness[:, u]) for u in range(Archive_fitness.shape[1])])
        min_fitness = np.array([np.min(Archive_fitness[:, u]) for u in range(Archive_fitness.shape[1])])
        epsilon = (max_fitness - min_fitness) / (3 * len(Archive))
        for j in range(len(new_Archive)):
            a = new_Archive[j]
            if dominates(s, a):
                sets_of_s_dominates.append(j)
            elif dominates(a, s):
                number_of_dominates_s += 1
            if epsilon_dominates(a, s, epsilon):
                number_of_epsilon_dominates_s += 1
        # Step 2: If n != 0 (i.e. exists solutions in Archive that dominate s), then s is discarded
        # 步骤 2：如果 n != 0（即 Archive 中存在解能支配 s），则丢弃 s
        if number_of_dominates_s != 0:
            continue
        # Step 3: If C is not empty (i.e. exists solutions in Archive that s dominates)
        # Then remove them from Archive, and add s to Archive
        # 步骤 3：如果 C 不为空（即 Archive 中存在解能被 s 支配），则从 Archive 中删除它们，并将 s 添加到 Archive
        if len(sets_of_s_dominates) != 0:
            new_Archive = np.delete(new_Archive, sets_of_s_dominates, axis=0)
            new_Archive = np.concatenate([new_Archive, [s]])
        # Step 4: Otherwise, if ne != 0 (i.e. exists solutions in Archive that epsilon-dominate s), then s is discarded
        # 步骤 4：否则，如果 ne != 0（即 Archive 中存在解能以 epsilon 支配解 s），则丢弃 s
        elif number_of_epsilon_dominates_s != 0:
            continue
        # Step 5: Otherwise, if ne == 0 (i.e. no solutions in Archive epsilon-dominate s), then add s to Archive
        # 步骤 5：否则，如果 ne == 0（即 Archive 中不存在解能以 epsilon 支配解 s），则将 s 添加到 Archive
        else:
            new_Archive = np.concatenate([new_Archive, [s]])
    print('  Archive end length: ' + color('%d' % len(new_Archive), 'g'))
    # Step 6: If |Archive| > N, then shear Archive until |Archive| = N
    # 步骤 6：如果 |Archive| > N，则修剪 Archive，直到 |Archive| = N
    if len(new_Archive) > max_arch:
        print('  len(new_Archive) > N,', color('start shear Archive...', 'c'))
        new_Archive = shear_Archive(new_Archive, max_arch)
    return new_Archive


# Shear Archive if its size is larger than max_arch
# 修剪 Archive，使其大小不超过 max_arch
def shear_Archive(Archive, max_arch):
    new_Archive = Archive.copy()
    # Initial epsilon is twice the original
    # 初始 epsilon 是原始 epsilon 的两倍
    Archive_fitness = Archive[:, 4:]
    max_fitness = np.array([np.max(Archive_fitness[:, u]) for u in range(Archive_fitness.shape[1])])
    min_fitness = np.array([np.min(Archive_fitness[:, u]) for u in range(Archive_fitness.shape[1])])
    epsilon = (max_fitness - min_fitness) / (3 * len(Archive))
    epsilon *= 2

    # Use epsilon-dominance to shear Archive
    # 用 epsilon 支配来修剪 Archive
    def shear(Archive, e):
        new_Archive = []
        origin_Archive = Archive.copy()
        while len(origin_Archive) > 0:
            a = origin_Archive[0]
            origin_Archive = np.delete(origin_Archive, 0, axis=0)
            num_of_epsilon_dominates_a = 0
            for b in origin_Archive:
                if epsilon_dominates(b, a, e):
                    num_of_epsilon_dominates_a += 1
            if num_of_epsilon_dominates_a == 0:
                new_Archive.append(a)
        return np.array(new_Archive)

    # Shear Archive until its size is smaller than N, otherwise double epsilon and shear again
    # 修剪 Archive，直到其大小小于 N，否则将 epsilon 加倍，再次修剪
    while len(new_Archive) > max_arch:
        new_Archive = shear(new_Archive, epsilon)
        epsilon *= 2
    return new_Archive


# Test
if __name__ == '__main__':
    solutions1 = np.array([[1, 2, 3], [1, 3, 2], [1, 1, 4], [2, 3, 4]])
    solutions2 = np.array(
        [[5, 6, 7], [2, 3, 10], [4, 4, 8], [1, 4, 15], [2, 5, 14], [4, 7, 12], [0, 5, 20], [4, 5, 13]])
    fitness = lambda x: np.array([np.sum(x), np.max(x)])
    front_layers1 = fast_non_dominated_sort(solutions1)
    front_layers2 = fast_non_dominated_sort(solutions2)
    print(get_sorted_new_population(solutions1))
    # print(get_new_population(solutions2, front_layers2, use_fitness=True, fitness=fitness))
