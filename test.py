import numpy as np
from sklearn import preprocessing
from numpy.linalg import inv
import time
import math

A = np.array([[1.,2.],[2.,2.]])

hold = np.square(A)

average = 0
d_hold = []
for i in range(len(hold)):
	for j in range(len(hold[i])):
		average += hold[i][j]
	d_hold.append(average/len(hold[i]))
	average = 0

print d_hold