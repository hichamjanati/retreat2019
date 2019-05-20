import numpy as np


def save_name(k):
    if type(k) is str:
        save_string = k
    else:
        save_string = '_'.join([str(x) for x in k])


def save_dict(d):
    expe_name = d['expe']
    del d['expe']
    for k in d:
        res = d[k]
        save_string = save_name(d)
        np.savetxt('results/' + expe_name + save_string + '.csv', res)
