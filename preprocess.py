import numpy as np

import mne
import pandas as pd
from joblib import Memory


mem = Memory(location='.', verbose=0)

n_subj = 640
p = 306


@mem.cache()
def get_covs_and_ages(subjects, scale=1e22):
    data = mne.externals.h5io.read_hdf5('data/covs_allch_oas.float32.h5')
    probs = [241, 300, 327]
    for prob in probs[::-1]:
        del data[prob]
    covs = [scale * data[subject]['covs'] for subject in subjects]
    ids = [data[subject]['subject'] for subject in subjects]
    df = pd.read_csv('data/participants.csv')
    ages = [int(df[df['Observations'] == subj]['age']) for subj in ids]
    return np.array(covs), ages


@mem.cache()
def get_mean(subjects):
    X, y = get_covs_and_ages(subjects)
    C = X.reshape(9 * n_subj, p, p)
    return np.mean(C, axis=0)


@mem.cache()
def project_common_space(subjects, rank=65):
    C = get_mean(subjects)
    d, V = np.linalg.eigh(C)
    d = d[::-1]
    V = V[:, ::-1]
    proj_mat = V[:, :rank].T
    X, y = get_covs_and_ages(subjects)
    n_subj, _, _, _ = X.shape
    op = np.zeros((n_subj, 9, rank, rank))
    for subject in subjects:
        for j in range(9):
            op[subject, j] = proj_mat.dot(X[subject, j]).dot(proj_mat.T)
    return op, y


@mem.cache()
def project_own_space(subjects, rank=65):
    X, y = get_covs_and_ages(subjects)
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


if __name__ == '__main__':
    subjects = np.arange(640)
    project_own_space(subjects)
    project_common_space(subjects)
