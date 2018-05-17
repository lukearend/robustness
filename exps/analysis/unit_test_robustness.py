"""Train a model."""

import os
import sys
import time
import argparse
import pickle

import numpy as np

import tensorflow as tf

append_path = {
    '/raid': '/raid/poggio/home/larend/robust/src',
    '/om': '/om/user/larend/robust/src',
    '/cbcl': '/cbcl/cbcl01/larend/robust/src'}['/cbcl']
sys.path.append(append_path)
import estimator


def main():
    num_filters = {
        0.25: 4,
        0.5: 8,
        1: 16,
        2: 32,
        4: 64}[1]

    base_data_dir = {
        '/raid': '/raid/poggio/home/larend/data',
        '/om': '/om/user/larend/data',
        '/cbcl': '/cbcl/cbcl01/larend/data'}['/cbcl']

    data_dir = {
        'cifar10': '{}/cifar-10-tfrecords'.format(base_data_dir),
        'imagenet': '{}/imagenet-tfrecords'.format(base_data_dir)}['cifar10']

    for crossval in range(1):
        print('crossval: {}'.format(crossval))
        results = []

        model = estimator.Estimator(
            model_dir='/cbcl/cbcl01/larend/models/robust/cifar10/00002',
            params={
                'batch_size': 100,
                'dataset': 'cifar10',
                'use_batch_norm': True,
                'num_filters': num_filters},
            tf_random_seed=int(time.time()))

        for k, perturbation_type in enumerate([1]):
            print('perturbation: {}'.format(perturbation_type))

            perturbation_amounts = {
                0: np.linspace(0.0, 1.0, 7),
                1: np.linspace(0.0, 1.0, 7),
                2: np.linspace(0.0, 1.0, 7)}[perturbation_type]

            results.append([np.zeros(len(perturbation_amounts)) for _ in range(2)])
            for i, perturbation_amount in enumerate([perturbation_amounts[3]]):
                kill_mask = [None for _ in range(19)]

                for j, split in enumerate(['validation', 'train']):
                    print('split: {}'.format(split))

                    t_0 = time.time()
                    accuracy = model.robustness(
                        perturbation_type,
                        perturbation_amount,
                        kill_mask,
                        data_dir=data_dir,
                        split=split)
                    t_1 = time.time()

                    print('accuracy: {}'.format(accuracy))
                    print('time: {}'.format(t_1 - t_0))

                    results[k][j][i] = accuracy

        out_dir = '/cbcl/cbcl01/larend/tmp'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with open(os.path.join(out_dir, 'robustness{}.pkl'.format(crossval)), 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

        tf.reset_default_graph()

    print('done :)')


if __name__ == '__main__':
    main()
