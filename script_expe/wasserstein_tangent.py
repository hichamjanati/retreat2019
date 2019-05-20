import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class TangentSpace(BaseEstimator, TransformerMixin):
    def __init__(self, rank, ref=None):
        """Init."""
        self.rank = rank
        self.ref = ref

    def fit(self, X, y=None):
        ref = self.ref
        if ref is None:
            # ref = mean_covs(X, rank=self.rank)
            ref = np.mean(X, axis=0)
        Y = to_quotient(ref, self.rank)
        self.reference_ = ref
        self.Y_ref_ = Y
        return self

    def transform(self, X, verbose=False):
        n_mat, n, _ = X.shape
        output = np.zeros((n_mat, n * self.rank))
        for j, C in enumerate(X):
            if verbose:
                print('\r %d / %d' % (j+1, n_mat), end='', flush=True)
            Y = to_quotient(C, self.rank)
            output[j] = logarithm_(Y, self.Y_ref_).ravel()
        return output


def to_quotient(C, rank):
    d, U = np.linalg.eigh(C)
    U = U[:, -rank:]
    d = d[-rank:]
    Y = U * np.sqrt(d)
    return Y


def distance2(S1, S2, rank=None):
    Sq = sqrtm(S1, rank)
    P = sqrtm(np.dot(Sq, np.dot(S2, Sq)), rank)
    return np.trace(S1) + np.trace(S2) - 2 * np.trace(P)


def mean_covs(covmats, rank, tol=10e-4, maxiter=50, init=None,
              sample_weight=None):
    Nt, Ne, Ne = covmats.shape
    if sample_weight is None:
        sample_weight = np.ones(Nt)
    if init is None:
        C = np.mean(covmats, axis=0)
    else:
        C = init
    k = 0
    K = sqrtm(C, rank)
    crit = np.finfo(np.float64).max
    # stop when J<10^-9 or max iteration = 50
    while (crit > tol) and (k < maxiter):
        k = k + 1

        J = np.zeros((Ne, Ne))

        for index, Ci in enumerate(covmats):
            tmp = np.dot(np.dot(K, Ci), K)
            J += sample_weight[index] * sqrtm(tmp)

        Knew = sqrtm(J, rank)
        crit = np.linalg.norm(Knew - K, ord='fro')
        K = Knew
    if k == maxiter:
        print('Max iter reach')
    C = np.dot(K, K)
    return C


def sqrtm(C, rank=None):
    if rank is None:
        rank = C.shape[0]
    d, U = np.linalg.eigh(C)
    U = U[:, -rank:]
    d = d[-rank:]
    return np.dot(U, np.sqrt(np.abs(d))[:, None] * U.T)


def logarithm_(Y, Y_ref):
    prod = np.dot(Y_ref.T, Y)
    U, D, V = np.linalg.svd(prod, full_matrices=False)
    Q = np.dot(U, V).T
    return np.dot(Y, Q) - Y_ref


if __name__ == '__main__':
    rng = np.random.RandomState(0)
    n_mat = 10
    n = 10
    p = 5
    eps = 1e-2
    Y = rng.randn(n, p)
    C_ref = Y.dot(Y.T)
    X = np.zeros((n_mat, n, n))
    for i in range(n_mat):
        Y_ = Y + eps * rng.randn(n, p)
        X[i] = Y_.dot(Y_.T)
    ts = TangentSpace(p)
    ts.fit(X, ref=C_ref)
    X_t = ts.transform(X)
    D_m = np.zeros(n_mat)
    D_T = np.zeros(n_mat)
    for i in range(n_mat):
        D_m[i] = X_t[i].dot(X_t[i])
        D_T[i] = distance2(X[i], C_ref, p)
    print(np.mean((D_m - D_T) / D_m.max()))
