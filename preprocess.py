import numpy as np

from scipy.linalg import eigh
import mne
import pandas as pd
from joblib import Memory
from pyriemann.tangentspace import TangentSpace
from sym_rank import TangentSpace as TangentSpace_
from common_space import project_space as pcs


mem = Memory(location='.', verbose=0)


@mem.cache()
def get_covs_and_ages(subjects, scale=1e22, picks='all'):
    data = mne.externals.h5io.read_hdf5('data/covs_allch_oas.float32.h5')
    probs = [241, 300, 327]
    for prob in probs[::-1]:
        del data[prob]
    covs = []
    for subject in subjects:
        c = scale * data[subject]['covs']
        if picks == 'mag':
            c = c[:, 2::3, :][:, :, 2::3]
        elif picks == 'grad':
            m = np.arange(306).reshape(102, 3)[:, :2].ravel()
            c = c[:, m, :][:, :, m]
        covs.append(c)
    ids = [data[subject]['subject'] for subject in subjects]
    df = pd.read_csv('data/participants.csv')
    ages = [int(df[df['Observations'] == subj]['age']) for subj in ids]
    return np.array(covs), np.array(ages)


def shrinkage(C, alpha):
    n, _ = C.shape
    return (1 - alpha) * C + alpha * np.trace(C) * np.eye(n) / n


@mem.cache()
def spoc(subjects, rank=40, picks='all'):
    print('spoc')
    X, y = get_covs_and_ages(subjects, picks=picks)
    n_s, n_f, p, _ = X.shape
    X_o = np.zeros((n_s, n_f, rank, rank))
    y_ = y - y.mean()
    y_ /= np.std(y_)
    for i in range(n_f):
        Cw = np.mean(y_[:, None, None] * X[:, i], axis=0)
        C = np.mean(X[:, i], axis=0)
        C = shrinkage(C, 0.5)
        Cw = shrinkage(C, 0.5)
        eigvals, eigvecs = eigh(Cw, C)
        eigvals = eigvals.real
        eigvecs = eigvecs.real
        order = np.argsort(np.abs(eigvals))[::-1][:rank]
        proj_mat = eigvecs[:, order].T
        for subject in subjects:
            X_o[subject, i] = proj_mat.dot(X[subject, i]).dot(proj_mat.T)
    return X_o, y


@mem.cache()
def get_mean(subjects, picks='all'):
    X, y = get_covs_and_ages(subjects, picks=picks)
    _, _, p, _ = X.shape
    C = X.reshape(9 * len(subjects), p, p)
    return np.mean(C, axis=0)


@mem.cache()
def project_common_space(subjects, rank=65, picks='all'):

    C = get_mean(subjects, picks=picks)
    d, V = np.linalg.eigh(C)
    d = d[::-1]
    V = V[:, ::-1]
    proj_mat = V[:, :rank].T
    X, y = get_covs_and_ages(subjects, picks=picks)
    print("projecting in the common space")
    n_subj, _, _, _ = X.shape
    op = np.zeros((n_subj, 9, rank, rank))
    for subject in subjects:
        for j in range(9):
            op[subject, j] = proj_mat.dot(X[subject, j]).dot(proj_mat.T)
    return op, y


@mem.cache()
def project_own_space(subjects, rank=65, picks='all'):

    X, y = get_covs_and_ages(subjects, picks=picks)
    print("projecting in the own space")
    n_subj, _, _, _ = X.shape
    op = np.zeros((n_subj, 9, rank, rank))
    for subject in subjects:
        C = np.mean(X[subject], axis=0)
        d, V = np.linalg.eigh(C)
        d = d[::-1]
        V = V[:, ::-1]
        proj_mat = V[:, :rank].T
        for j in range(9):
            op[subject, j] = proj_mat.dot(X[subject, j]).dot(proj_mat.T)
    return op, y


def project_covariances(subjects, rank=65, picks="all", mode="common"):
    return {'common': project_common_space,
            'own': project_own_space}[mode](subjects, rank, picks=picks)


@mem.cache()
def project_tangent_space(subjects, rank=65, picks="all", mode="common",
                          reg=1e-6):
    if mode == "common":
        X, y = project_common_space(subjects, rank, picks)
    elif mode == 'own':
        X, y = project_common_space(subjects, rank, picks)
    elif mode == 'spoc':
        X, y = spoc(subjects, rank, picks)
    elif mode == "csf":
        X, y = get_covs_and_ages(subjects, picks=picks)
        X = pcs(X, rank, common_f=True)
    elif mode == "cs":
        X, y = get_covs_and_ages(subjects, picks=picks)
        X = pcs(X, rank, common_f=False)
    else:
        X, y = get_covs_and_ages(subjects, picks=picks)
    print("projecting in the tangent space")
    n_subj, n_freqs, p, _ = X.shape
    if reg:
        for i in range(n_subj):
            for f in range(n_freqs):
                X[i, f] += reg * np.eye(p)
    ts = np.zeros((n_subj, n_freqs, int(p * (p+1)/2)))
    n_s_train = 100
    for f in range(n_freqs):
        sl = np.random.permutation(np.arange(640))[:n_s_train]
        ts[:, f, :] = TangentSpace().fit(
                        X[sl, f, :, :]).transform(X[:, f, :, :])
    return ts, y


@mem.cache()
def project_tangent_space_rank(subjects, rank=65, picks="all"):
    X, y = get_covs_and_ages(subjects, picks=picks)
    print("projecting in the tangent space")
    n_subj, n_freqs, p, _ = X.shape
    ts = np.zeros((n_subj, n_freqs, p * rank))
    n_s_train = 100
    for f in range(n_freqs):
        sl = np.random.permutation(np.arange(640))[:n_s_train]
        ts[:, f, :] = TangentSpace_(rank).fit(
                        X[sl, f, :, :]).transform(X[:, f, :, :], verbose=True)
    return ts, y


if __name__ == '__main__':
    subjects = np.arange(640)
    project_own_space(subjects, picks='mag')
    project_common_space(subjects, picks='mag')
