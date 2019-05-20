import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from generation import generate_covariances
from sklearn.pipeline import Pipeline
from pyriemann.tangentspace import TangentSpace
from wasserstein_tangent import TangentSpace as WT
from sklearn_classes import logDiag, Random, Ravel
from sklearn.model_selection import cross_val_score

from utils import save_dict


distances = np.linspace(0.001, 4, 4)
results = {'expe': 'synth_da', 'distances': distances}


rng = 5

n_matrices = 100
n_channels = 5
n_sources = 5
sigma = 0

direction_A = np.random.RandomState(4).randn(n_channels, n_channels)
# direction_A -= direction_A.T
direction_A /= np.linalg.norm(direction_A)

embeddings = [Random(), logDiag(), WT(n_channels, ref=None), TangentSpace(),
              Ravel()]
names = ['Chance level', 'Log-powers', 'Wasserstein', 'Geometric', 'Euclid']
for distance_A_id in distances:
    X, y = generate_covariances(n_matrices, n_channels, n_sources, sigma=sigma,
                                distance_A_id=distance_A_id, log=False,
                                direction_A=direction_A, rng=rng)
    for name, embedding in zip(names, embeddings):
        print(name)
        lr = RidgeCV(alphas=np.logspace(-7, 3, 10),
                     scoring='neg_mean_absolute_error')
        if name == 'Chance level':
            pipeline = Pipeline([('emb', logDiag()),
                                 ('sc', StandardScaler()),
                                 ('lr', DummyRegressor())])
        else:
            pipeline = Pipeline([('emb', embedding),
                                 ('sc', StandardScaler()),
                                 ('lr', lr)])

        sc = cross_val_score(pipeline, X, y,
                             scoring='neg_mean_absolute_error',
                             cv=10, n_jobs=3)
        results[(name, distance_A_id)] = - np.mean(sc)
        # print(sc)
# save_dict(results)


f, ax = plt.subplots(figsize=(4, 3))
# chance_levels = []
# for distance in distances:
#     chance_levels.append(results[('Chance level', distance)])
# for i, distance in enumerate(distances):
#     for name in names:
#         results[(name, distance)] /= chance_levels[i]
for name in names:
    if name != 'Chance level':
        ls = None
    else:
        ls = '--'
    ax.plot(distances, [results[(name, distance)] for distance in distances],
            label=name,
            linewidth=2,
            linestyle=ls)
ax.legend(loc='lower right')
lblx = ax.set_xlabel(r'Distance between $A$ and $I_P$')
# ax.set_xlim(0.5, len(distances) + .5)
# ax.set_yscale('log')
plt.grid()
lbl = ax.set_ylabel(r'Normalized M.A.E.')

plt.savefig('figs/da.pdf', bbox_extra_artists=[lbl, lblx],
            bbox_inches='tight')
plt.show()
