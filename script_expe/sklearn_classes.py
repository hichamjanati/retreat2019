import numpy as np

from sklearn.base import TransformerMixin


class Random(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.random.randn(X.shape[0], 1)


class logDiag(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.log(np.diagonal(X, axis1=1, axis2=2))


class Ravel(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.reshape(X.shape[0], -1)
