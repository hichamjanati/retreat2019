
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from preprocess import (project_own_space, project_common_space,
                        get_covs_and_ages)

seed = 42
n_samples = 640
subjects = np.arange(n_samples)
X, y = get_covs_and_ages(subjects, picks='mag')


def extract_features(X, freq_index=0):
    n_features = X.shape[-1]
    ii = np.arange(n_features)
    return np.log(X[:, freq_index].reshape(len(X), -1) + 1e-1)


fts = [("ft%s" % k, make_pipeline(
        FunctionTransformer(extract_features, validate=False,
                            kw_args=dict(freq_index=k)),
        )) for k in range(9)]

cv = KFold(n_splits=10, shuffle=True, random_state=seed)
ridge = make_pipeline(FeatureUnion(fts), StandardScaler(),
                      RidgeCV(alphas=np.logspace(-3, 5, 100)))
score = cross_val_score(ridge, X, y, cv=cv,
                        scoring="neg_mean_absolute_error")
print(f"Mean MAE RidgeCV = {- score.mean()}")


# lasso = make_pipeline(ts, Lasso())
