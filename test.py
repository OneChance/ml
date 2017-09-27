import numpy as np

a = np.ones([2, 2, 2])
a = a.reshape(-1, 2 * 2 * 2)
print(a)
