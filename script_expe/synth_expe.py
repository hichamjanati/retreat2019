import numpy as np
import matplotlib.pyplot as plt
from pylab import plot, show, savefig, xlim, figure, \
                  ylim, legend, boxplot, setp, axes

from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.base import TransformerMixin
from pyriemann.tangentspace import TangentSpace
from wasserstein_tangent import TangentSpace as WT


rng = np.random.RandomState(0)

metrics = ['riemann', 'euclid', 'wasserstein', 'random', 'diag', 'logdiag']
names = ['Geometric', 'Euclidian', 'Wasserstein',  'Random',
         'Powers', 'log-Powers']
colors = ['indianred', 'cornflowerblue', 'green', 'black', 'orange', 'gold']
sigmas = np.power(10., np.arange(-3, 3))
n_subjects = 100
p = 10
n_sources = 3
scores = {}


for sigma in sigmas:
    # sklearn pipeline
    for metric in metrics:
        if metric == 'random':
            ts = Random()
        elif metric == 'signal':
            ts = Ravel()
        elif metric == 'diag':
            ts = Diag()
        elif metric == 'logdiag':
            ts = logDiag()
        elif metric == 'riemann':
            ts = TangentSpace()
        elif metric == 'wasserstein':
            ts = WT(p)
        elif metric == 'euclid':
            ts = Ravel()

        lr = RidgeCV(alphas=np.logspace(-8, 3, 100))
        pipe = Pipeline([('tangent', ts),
                         ('lstq', lr)])
        sc = cross_val_score(pipe, X, y,
                             scoring='neg_mean_absolute_error',
                             cv=5)
        scores[(sigma, metric)] = -sc


def setBoxColors(bp):
    for i, color in enumerate(colors):
        setp(bp['boxes'][i], color=color)
        setp(bp['caps'][2 * i], color=color)
        setp(bp['caps'][2 * i + 1], color=color)
        setp(bp['whiskers'][2 * i], color=color)
        setp(bp['whiskers'][2 * i + 1], color=color)
        setp(bp['medians'][i], color=color)


f, ax = plt.subplots(figsize=(4, 3))
x = np.arange(1, len(sigmas) + 1)
for sigma in sigmas:
    for metric in metrics:
        scores[(sigma, metric)] /= scores[(sigma, 'random')]
for color, metric, name in zip(colors, metrics, names):
    if metric != 'random':
        ls = None
    else:
        ls = '--'
    ax.plot(x, [np.mean(scores[(sigma, metric)]) for sigma in sigmas],
            label=name,
            color=color,
            linewidth=2,
            linestyle=ls)
ax.legend(ncol=2, loc='upper left')
ax.set_xticklabels([r'$10^{%d}$' % p for p in np.log10(sigmas)])
ax.set_xticks(x)
lblx = ax.set_xlabel(r'$\sigma$')
ax.set_xlim(0.5, len(sigmas) + .5)
ax.set_yscale('log')
plt.grid()
lbl = ax.set_ylabel(r'Normalized M.A.E.')

# plt.savefig('figs/synth_expe.pdf', bbox_extra_artists=[lbl, lblx],
#             bbox_inches='tight')
plt.show()
