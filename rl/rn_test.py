import numpy as np
import tensorflow as tf

a = [0, 0, 0, 0, 1]
a -= np.mean(a)
a /= np.std(a)
print(a)
