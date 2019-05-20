
import numpy as np
from sklearn.linear_model import RidgeCV, LinearRegression
from celer import LassoCV
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from preprocess import (project_own_space, project_common_space,
                        project_tangent_space, get_covs_and_ages, spoc,
                        project_tangent_space_rank)

from time import time
from joblib import Memory

from sklearn.svm import LinearSVR
from mlxtend.regressor import StackingRegressor

mem = Memory(location='.', verbose=0)

seed = 42
n_samples = 640
subjects = np.arange(n_samples)
X, y = project_tangent_space(subjects, picks="mag", rank=70, reg=1e-7, mode="cs")
# X, y = project_tangent_space_rank(subjects, picks="mag", rank=70)
# X, y = spoc(subjects, picks='mag', rank=70)
# X, y = get_covs_and_ages(subjects, picks='mag')
# X = pts(X, 60)
# X, y = project_common_space(subjects, picks='mag', rank=70)
# n_samples, n_f, n_p, _ = X.shape
# X = np.log(np.diagonal(X, axis1=2, axis2=3))
# X_ = np.zeros((n_samples, n_f, int(n_p * (n_p+1)/2)))
# idx = np.tril_indices(n_p)
# for i in range(n_samples):
#     for j in range(n_f):
#         X_[i, j] = X[i, j][idx]

X = X.reshape(n_samples, -1)
# X = X[:100]
# y = y[:100]

#
# def extract_features(X, freq_index=0):
#     # print(X.shape)
#     # Y = X.copy().reshape(n_samples, n_f, n_p)
#     return X[:, freq_index, :]
#
#
# regressors = [make_pipeline(FunctionTransformer(extract_features,
#                                                 validate=False,
#                                                 kw_args=dict(freq_index=k)),
#                             StandardScaler(),
#                             RidgeCV(alphas=np.logspace(-3, 5, 100)))
#               for k in range(9)]
#
# lr = RandomForestRegressor(n_estimators=500)
#
# stregr = StackingRegressor(regressors=regressors,
#                            meta_regressor=lr)

cv = RepeatedKFold(n_repeats=1, n_splits=10, random_state=seed)
ridge = make_pipeline(StandardScaler(),
                      RidgeCV(alphas=np.logspace(-3, 5, 100)))
# ridge = make_pipeline(StandardScaler(),
#                       LassoCV(n_alphas=10))
t = time()
score = cross_val_score(ridge, X, y, cv=cv,
                        scoring="neg_mean_absolute_error", n_jobs=3,
                        verbose=True)
print(f"Mean MAE = {- score.mean()}")
print(f"run in {time() - t} seconds")
ridge.fit(X, y)
print(ridge.steps[1][1].alpha_)
