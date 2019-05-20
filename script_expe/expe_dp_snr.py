import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from generation import generate_covariances
from sklearn.pipeline import Pipeline
from pyriemann.tangentspace import TangentSpace
from wasserstein_tangent import TangentSpace as WT
from sklearn_classes import logDiag, Random
from sklearn.model_selection import cross_val_score

from utils import save_dict


sigmas = [1e-2, 1e-1, 1, 10, 100]
# sigmas = [1, 10]
results = {'expe': 'synth_snr', 'sigmas': sigmas}


rng = 4

n_matrices = 100
n_channels = 5
n_sources = 2
distance_A_id = 10

embeddings = [Random(), logDiag(), WT(n_channels), TangentSpace()]
names = ['Chance level', 'Log-powers', 'Wasserstein', 'Geometric']
for sigma in sigmas:
    X, y = generate_covariances(n_matrices, n_channels, n_sources, sigma=sigma,
                                distance_A_id=distance_A_id, rng=rng)
    for name, embedding in zip(names, embeddings):
        print(name)
        lr = RidgeCV(alphas=np.logspace(-7, 3, 100),
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
        results[(name, sigma)] = -np.mean(sc)
# save_dict(results)


f, ax = plt.subplots(figsize=(4, 3))
x = np.arange(1, len(sigmas) + 1)
chance_levels = []
for sigma in sigmas:
    chance_levels.append(results[('Chance level', sigma)])
for i, sigma in enumerate(sigmas):
    for name in names:
        results[(name, sigma)] /= chance_levels[i]
for name in names:
    if name != 'Chance level':
        ls = None
    else:
        ls = '--'
    ax.plot(x, [results[(name, sigma)] for sigma in sigmas],
            label=name,
            linewidth=2,
            linestyle=ls)

ax.set_xticklabels([r'$10^{%d}$' % p for p in np.log10(sigmas)])
ax.set_xticks(x)
lblx = ax.set_xlabel(r'$\sigma$')
ax.set_xlim(0.5, len(sigmas) + .5)
# ax.set_yscale('log')
plt.grid()
lbl = ax.set_ylabel(r'Normalized M.A.E.')
ax.hlines(0, 1, len(sigmas), label=r'Perfect', color='k', linestyle='--')
ax.legend(loc='lower right')
ax.set_xlim(.9, len(sigmas) + .1)
plt.savefig('figs/snr.pdf', bbox_extra_artists=[lbl, lblx],
            bbox_inches='tight')
plt.show()
