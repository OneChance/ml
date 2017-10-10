import numpy as np
import tensorflow as tf

memory = np.zeros((3, 1 * 2 + 2))
memory[0] = np.hstack((1, [2, 3], 2))
memory[1] = np.hstack((2, [5, 6], 3))
memory[2] = np.hstack((3, [7, 8], 4))

a = memory[:, 1].astype(int)

print(a)
