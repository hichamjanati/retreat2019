import numpy as np
from sklearn.base import BaseEstimator
from nilearn.connectome import ConnectivityMeasure
from pyriemann.tangentspace import TangentSpace


class Identity(BaseEstimator):
    def fit(self, X, y=None):
        self.covariance_ = X
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


cm = ConnectivityMeasure(cov_estimator=Identity(), kind='tangent', vectorize=True)

X = np.random.randn(100, 3, 3)
for x in X:
    x[:] = x.dot(x.T)

XX = cm.fit_transform(X)
XXX = TangentSpace().fit_transform(X)
