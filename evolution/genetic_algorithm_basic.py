# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 10  # DNA长度
POP_SIZE = 100  # 繁衍一代的数量
CROSS_RATE = 0.8  # 交叉配对的几率
MUTATION_RATE = 0.003  # 变异几率
N_GENERATIONS = 200
X_BOUND = [0, 5]


# 模拟的生存环境,函数最大值为最优
def F(_x):
    return np.sin(10 * _x) * _x + np.cos(2 * _x) * _x


# 根据函数F值,确定物种适应能力,减去最小值是为了生成非负值
def get_fitness(pred):
    return pred + 1e-3 - np.min(pred)


# 将DNA序列转换成x值
def translateDNA(_pop):
    return _pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / (2 ** DNA_SIZE - 1) * X_BOUND[1]


# 适应能力越强,被选中的几率会越高
def select(_pop, _fitness):
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=_fitness / _fitness.sum())
    print("idx:", idx)
    return _pop[idx]


# 交叉配对
def crossover(_parent, _pop):
    # CROSS_RATE的几率会进行繁衍
    if np.random.rand() < CROSS_RATE:
        # 选择另一个当代物种
        i_ = np.random.randint(0, POP_SIZE, size=1)
        # 替换的DNA位[False  True  True False False  True False  True  True False]
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)
        # 替换,生成新物种
        _parent[cross_points] = _pop[i_, cross_points]
    return _parent


# 变异
def mutate(_child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            _child[point] = 1 if _child[point] == 0 else 0
    return _child


# 初始化DNA,DNA完全一样的物种
pop = np.random.randint(0, 2, (1, DNA_SIZE)).repeat(POP_SIZE, axis=0)

plt.ion()
x = np.linspace(X_BOUND[0], X_BOUND[1], 200)
plt.plot(x, F(x))

for _ in range(N_GENERATIONS):
    F_values = F(translateDNA(pop))

    if 'sca' in globals():
        sca.remove()
    sca = plt.scatter(translateDNA(pop), F_values, s=200, lw=0, c='red', alpha=0.5)
    plt.pause(0.05)

    fitness = get_fitness(F_values)

    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child

    print("第: ", _, "代繁衍结束,最适合的DNA:", pop[np.argmax(fitness), :])

plt.ioff()
plt.show()
