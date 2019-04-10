
import numpy as np
from sklearn.linear_model import RidgeCV
from celer import LassoCV
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from preprocess import (project_own_space, project_common_space,
                        project_tangent_space, get_covs_and_ages)

from time import time
from joblib import Memory


mem = Memory(location='.', verbose=0)

seed = 42
n_samples = 640
subjects = np.arange(n_samples)
X, y = project_tangent_space(subjects, picks="mag", rank=65, reg=1e-6,
                             mode='common')
X = X.reshape(len(X), -1)
# X = X[:100]
# y = y[:100]


# @mem.cache()
# def get_tangent(X):
#     return TangentSpace().fit_transform(X)
#


# fts = [("ft%s" % k, make_pipeline(
#         FunctionTransformer(extract_features, validate=False,
#                             kw_args=dict(freq_index=k)),
#         FunctionTransformer(get_tangeant)))
#        for k in range(9)]


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
