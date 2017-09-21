import numpy as np
import math

X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[0,1,1,0]]).T

syn0 = 2*np.random.random((3,4))-1
syn1 = 2*np.random.random((4,1))-1

print(np.dot(X,syn0))
