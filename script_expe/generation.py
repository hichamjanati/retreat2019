import numpy as np

from scipy.linalg import expm

from sklearn.utils import check_random_state


def generate_covariances(n_matrices, n_channels, n_sources, rank=None,
                         distance_A_id=0., direction_A=None,
                         distance_projs=0., sigma=0, log=True, rng=0):
    rng = check_random_state(rng)
    project = rank is not None
    # Generate A close from id
    if direction_A is None:
        direction_A = rng.randn(n_channels, n_channels)
    A = expm(distance_A_id * direction_A)
    print(A)
    # A = np.linalg.svd(A)[0]
    # Generate powers
    powers = rng.rand(n_matrices, n_sources)
    # Generate source covariances
    Cs = np.zeros((n_matrices, n_channels, n_channels))
    for i in range(n_matrices):
        Cs[i, :n_sources, :n_sources] = np.diag(powers[i])
        N_i = rng.randn(n_channels - n_sources, n_channels - n_sources)
        Cs[i, n_sources:, n_sources:] = N_i.dot(N_i.T)
    # Generate covariances X:
    X = np.array([A.dot(cs).dot(A.T) for cs in Cs])
    if project:  # Generate random projection of rank r
        W_list = []
        M = rng.randn(n_channels, n_channels)
        for _ in range(n_matrices):
            M_ = M + distance_projs * rng.randn(n_channels, n_channels)
            U, D, V = np.linalg.svd(M_)
            W_list.append(U[:, :rank].dot(D[:rank, None] * V[:rank, :]))
        X = np.array([W.dot(x).dot(W.T) for x in X])
    # Generate y
    alpha = rng.randn(n_sources)
    if log:
        y = np.log(powers).dot(alpha)
    else:
        y = (powers).dot(alpha)
    y += sigma * rng.randn(n_matrices)
    return X, y
