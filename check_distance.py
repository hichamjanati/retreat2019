import numpy as np
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.distance import distance

from joblib import Memory
from preprocess import project_covariances, project_tangent_space

mem = Memory(location='.', verbose=0)
@mem.cache()
def check_distance(subjects, rank=65, picks="all", mode="common",
                   reg=True):
    C, _ = project_covariances(np.arange(640), rank, picks,
                               mode)
    X, _ = project_tangent_space(np.arange(640), rank, picks,
                                 mode, reg)
    C = C[subjects]
    X = X[subjects]
    n_s, n_f, p, _ = C.shape
    D = np.zeros((n_s, n_s, n_f))
    D2 = np.zeros((n_s, n_s, n_f))
    for i in range(n_f):
        for j in range(n_s):
            for k in range(j+1):
                d = distance(C[j, i], C[k, i])
                D[j, k, i] = d
                D[k, j, i] = d
                d2 = np.linalg.norm(X[j, i] - X[k, i])
                D2[j, k, i] = d2
                D2[k, j, i] = d2
    return D, D2


if __name__ == '__main__':
    subjects = np.random.permutation(np.arange(640))[:10]
    D, D2 = check_distance(subjects, 30, 'mag', reg=False)
