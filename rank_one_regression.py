import numpy as np

from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.model_selection import cross_val_score, RepeatedKFold, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from preprocess import (project_own_space, project_common_space,
                        project_tangent_space, get_covs_and_ages)

from sklearn.utils import check_random_state
from time import time
from joblib import Memory

from scipy.optimize import check_grad, fmin_l_bfgs_b

from sklearn.base import BaseEstimator, RegressorMixin


def loss(alpha, beta, X, y, lbda, mu):
    n, = y.shape
    fit = 0.5 * np.sum((X.dot(alpha).dot(beta) - y) ** 2) / n
    na = np.sum(alpha ** 2)
    nb = np.sum(beta ** 2)
    return fit + lbda / 2 * na + mu / 2 * nb


def gradient_and_loss(alpha, beta, X, y, lbda, mu):
    n = len(y)
    Xa = X.dot(alpha)
    Xb = beta.dot(X)
    res = Xa.dot(beta) - y
    fit = 0.5 * np.sum(res ** 2) / n
    na = np.sum(alpha ** 2)
    nb = np.sum(beta ** 2)
    l_v = fit + lbda / 2 * na + mu / 2 * nb
    grad_a = res.dot(Xb) / n + lbda * alpha
    grad_b = res.dot(Xa) / n + mu * beta
    return l_v, grad_a, grad_b


def func(x, X, y, lbda, mu):
    n, p, q = X.shape
    l_v, g_a, g_b = gradient_and_loss(x[:q], x[q:], X, y, lbda, mu)
    return l_v, np.concatenate((g_a, g_b))


def fit(alpha, beta, X, y, lbda, mu):
    _, _, q = X.shape
    x = np.concatenate((alpha, beta))
    x, _, _ = fmin_l_bfgs_b(func, x, args=(X, y, lbda, mu))
    return x[:q], x[q:]


def alternate(alpha, beta, X, y, lbda, mu, max_iter=1000, tol=1e-5):
    n, p, q = X.shape
    for i in range(max_iter):
        # Beta
        aX = X.dot(alpha)
        G = aX.T.dot(aX) / n + mu * np.eye(p)
        beta = np.linalg.solve(G, y.dot(aX) / n)
        beta /= np.linalg.norm(beta)
        if i % 10 == 0 and i > 0:
            _, g, _ = gradient_and_loss(alpha, beta, X, y, lbda, mu)
            if np.max(np.abs(g)) < tol:
                break
        # Alpha
        bX = beta.dot(X)
        G = bX.dot(bX.T) / n + lbda * np.eye(n)
        alpha = bX.T.dot(np.linalg.solve(G, y)) / n
    return alpha, beta


class RoneRIDGE(BaseEstimator, RegressorMixin):
    def __init__(self, lbda=1., mu=1., rng=None):
        self.lbda = lbda
        self.mu = mu
        self.rng = check_random_state(rng)

    def fit(self, X, y, solver='alternate'):
        n, p, q = X.shape
        alpha0 = self.rng.randn(q)
        beta0 = self.rng.randn(p)
        if solver == 'alternate':
            alpha, beta = alternate(alpha0, beta0, X, np.array(y), self.lbda,
                                    self.mu, max_iter=100)
        else:
            alpha, beta = fit(alpha0, beta0, X, y, self.lbda, self.mu)
        self.alpha_ = alpha
        self.beta_ = beta
        return self

    def predict(self, X):
        return X.dot(self.alpha_).dot(self.beta_)


if __name__ == '__main__':
    seed = 42
    n_samples = 640
    subjects = np.arange(n_samples)
    X, y = project_tangent_space(subjects, picks="mag", rank=65, reg=1e-6,
                                 mode='common')
    # n, p, q = X.shape
    # X = X.reshape(len(X), -1)
    # X = StandardScaler().fit_transform(X)
    # X = X.reshape(n, p, q)
    # cv = KFold(n_splits=3, shuffle=True)
    # model = GridSearchCV(RoneRIDGE(rng=seed), cv=cv,
    #                      scoring='neg_mean_absolute_error',
    #                      param_grid={'lbda': np.logspace(-3, 5, 1),
    #                                  'mu': np.logspace(-3, 5, 2)},
    #                      verbose=1, n_jobs=3)
    # model.fit(X, y)
    # print(-model.best_score_)
