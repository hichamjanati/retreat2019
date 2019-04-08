import numpy as np

from pyriemann.utils.distance import distance
from joblib import Memory

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from preprocess import project_own_space, project_common_space

mem = Memory(location='.', verbose=0)


@mem.cache()
def compute_distances(subjects, rank=65, mode='common', metric='riemann'):
    print('computing projections')
    X, y = {'common': project_common_space,
            'own': project_own_space}[mode](subjects, rank)
    print('computing distances')
    n_subjects, n_freqs, p, _ = X.shape
    D = np.zeros((n_freqs, n_subjects, n_subjects))
    for freq in range(n_freqs):
        for i in range(n_subjects):
            print('\rFreq {}, done {}/{}'.format(freq, i+1, n_subjects,
                                                 end='', flush=True))
            for j in range(i+1):
                A = X[i, freq]
                B = X[j, freq]
                dist = distance(A, B, metric)
                D[freq, i, j] = dist
                D[freq, j, i] = dist
    return D, y


if __name__ == '__main__':
    subjects = np.arange(640)
    D, y = compute_distances(subjects)
    gamma = 0.1
    K = np.exp(-D * gamma)

    def kernel(i, j, f=3):
        return K[f, int(i), int(j)]
    krr = GridSearchCV(KernelRidge(kernel=kernel), cv=5,
                       param_grid={"alpha": np.logspace(-7, -3, 5)})
    # krr = KernelRidge(alpha=1, kernel=kernel, kernel_params={'gamma': gamma})
    cv = KFold(n_splits=10, shuffle=True)
    X = subjects.reshape(640, 1)
    cv_s = cross_val_score(krr, X,  y, cv=cv, n_jobs=2,
                           scoring='neg_mean_absolute_error')
