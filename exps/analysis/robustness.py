"""Train a model."""

import os
import sys
import time
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True)
parser.add_argument('--scale_factor', type=float, required=True)
parser.add_argument('--disable_batch_norm', dest='use_batch_norm',
                    action='store_false')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--host_filesystem', type=str, required=True)
FLAGS = parser.parse_args()

append_path = {
    '/raid': '/raid/poggio/home/larend/robust/src',
    '/om': '/om/user/larend/robust/src',
    '/cbcl': '/cbcl/cbcl01/larend/robust/src'}[FLAGS.host_filesystem]
sys.path.append(append_path)
import estimator


def main():
    if FLAGS.dataset == 'cifar10':
        num_filters = {
            0.25: 4,
            0.5: 8,
            1: 16,
            2: 32,
            4: 64}[FLAGS.scale_factor]
    else:
        num_filters = {
            0.25: 16,
            0.5: 32,
            1: 64,
            2: 128,
            4: 256}[FLAGS.scale_factor]

    data_dir = {
        'cifar10': '/om/user/larend/data/cifar-10-tfrecords',
        'imagenet': '/om/user/larend/data/imagenet-tfrecords'}[FLAGS.dataset]

    for crossval in range(3):
        results = [None for _ in range(5)]

        model = Estimator(
            model_dir=FLAGS.model_dir,
            params={
                'batch_size': 100,
                'dataset': FLAGS.dataset,
                'use_batch_norm': FLAGS.use_batch_norm,
                'num_filters': num_filters},
            tf_random_seed=int(time.time()))

        for perturbation_type in [0, 1, 2]:
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

                for j, split in enumerate(['validation', 'train']):
                    accuracy = model.robustness(
                        perturbation_type,
                        perturbation_amount,
                        kill_mask,
                        data_dir=data_dir,
                        split=split)

                    results[results_index][j][i] = accuracy

        if not os.path.exists(FLAGS.out_dir):
            os.makedirs(FLAGS.out_dir)

        with open(os.path.join(FLAGS.out_dir, 'robustness{}.pkl'.format(crossval)), 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

        tf.reset_default_graph()


if __name__ == '__main__':
    main()
