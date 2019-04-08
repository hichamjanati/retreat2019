import numpy as np

import mne
import pandas as pd
from joblib import Memory


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
    return np.array(covs), ages


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
    n_subj, _, _, _ = X.shape
    op = np.zeros((n_subj, 9, rank, rank))
    for subject in subjects:
        for j in range(9):
            op[subject, j] = proj_mat.dot(X[subject, j]).dot(proj_mat.T)
    return op, y


@mem.cache()
def project_own_space(subjects, rank=65, picks='all'):
    X, y = get_covs_and_ages(subjects, picks=picks)
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
    project_own_space(subjects, picks='mag')
    project_common_space(subjects, picks='mag')
