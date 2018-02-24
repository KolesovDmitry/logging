# -*- coding: utf-8 -*-

"""
Kernel density estimation
"""
import argparse


import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from data_reader import read_file, get_data

def main(flags):
    data = get_data(flags.data)

    plus = data.loc[(data['change'] == 1)]
    minus = data.loc[(data['change'] == 0)]

    names = [  # "current_slice",
        "blue", "blue_1",
        "green", "green_1",
        "red", "red_1",
        "nir", "nir_1",
        "swir1", "swir1_1",
        "swir2", "swir2_1"
    ]

    plus = np.array(plus[names])
    minus = np.array(minus[names])

    if minus.shape[0] > flags.sample_size:
        idx = np.random.randint(minus.shape[0], size=5000)
        minus_sample = minus[idx, :]
    else:
        minus_sample = minus

    if plus.shape[0] > FLAGS.sample_size:
        idx = np.random.randint(plus.shape[0], size=5000)
        plus_sample = plus[idx, :]
    else:
        plus_sample = plus


    grid = GridSearchCV(KernelDensity(),
                        {'bandwidth': np.linspace(0.001, 0.03, 30)},
                        n_jobs=-1,
                        cv=10) # 10-fold cross-validation
    grid.fit(plus_sample)
    print('Best bandwidth (plus):', grid.best_params_)
    kde_plus = grid.best_estimator_

    # import ipdb; ipdb.set_trace()

    grid.fit(minus_sample)
    print('Best bandwidth (minus):', grid.best_params_)
    kde_minus = grid.best_estimator_

    density_plus_p = kde_plus.score_samples(plus)
    density_plus_m = kde_plus.score_samples(minus)

    density_minus_p = kde_minus.score_samples(plus)
    density_minus_m = kde_minus.score_samples(minus)

    true_plus = density_plus_p > density_minus_p
    true_plus_prop =  1.0 * sum(true_plus.astype(np.int))/len(true_plus)

    true_minus = density_minus_m > density_plus_m
    true_minus_prop = 1.0 * sum(true_minus.astype(np.int)) / len(true_minus)

    print('True plus:', true_plus_prop)
    print('True minus:', true_minus_prop)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../../data/changes/expectedVSmedian/change_sample_*_seed*.csv',
                        help='Pattern for describe input data files')
    parser.add_argument('--sample_size', type=int, default='5000',
                        help='Size of sample')


    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
