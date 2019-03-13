import numpy as np

A = np.array([[0,-1],[2,3]])

evals, evecs = np.linalg.eig(A)

evecs = evecs*np.linalg.norm(evecs)

print(evals, "\n", evecs)
