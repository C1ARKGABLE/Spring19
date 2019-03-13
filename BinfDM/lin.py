import numpy as np


a = np.array([[0,-1],[2,3]])
print(a)
evals, evecs = np.linalg.eig(a)

print(evals,evecs)
