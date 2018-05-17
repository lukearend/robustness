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
        results = [None for _ in range(5)]

        model = estimator.Estimator(
            model_dir='/cbcl/cbcl01/larend/models/robust/cifar10/00002',
            params={
                'batch_size': 100,
                'dataset': 'cifar10',
                'use_batch_norm': True,
                'num_filters': num_filters},
            tf_random_seed=int(time.time()))

        for perturbation_type in [0, 1, 2]:
            print('perturbation: {}'.format(perturbation_type))

            perturbation_amounts = {
                0: np.linspace(0.0, 1.0, 7),
                1: np.linspace(0.0, 1.0, 7),
                2: np.linspace(0.0, 1.0, 7)}[perturbation_type]

            results_index = {
                0: 2,
                1: 3,
                2: 4}[perturbation_type]
            results[results_index] = [np.zeros(len(perturbation_amounts)) for _ in range(2)]
            for i, perturbation_amount in enumerate(perturbation_amounts):

                for j, split in enumerate(['train', 'validation']):
                    print('split: {}'.format(split))

                    # Build kernel file name.
                    pickle_dir = '/cbcl/cbcl01/larend/tmp'
                    split_str = {'train': '', 'validation': '_test'}[split]
                    kernel_filename = os.path.join(FLAGS.pickle_dir,
                                                   'kernel{}{}'.format(split_str, crossval))

                    t_0 = time.time()
                    accuracy = model.robustness(
                        perturbation_type,
                        perturbation_amount,
                        kernel_filename,
                        data_dir=data_dir,
                        split=split)
                    t_1 = time.time()

                    print('accuracy: {}'.format(accuracy))
                    print('time: {}'.format(t_1 - t_0))

                    results[results_index][j][i] = accuracy

        pickle_dir = '/cbcl/cbcl01/larend/tmp'
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)

        with open(os.path.join(pickle_dir, 'robustness{}.pkl'.format(crossval)), 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

        tf.reset_default_graph()

    print('done :)')
    print(results)


if __name__ == '__main__':
    main()
