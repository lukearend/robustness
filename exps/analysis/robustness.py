"""Train a model."""

import os
import sys
import time
import argparse
import pickle

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True)
parser.add_argument('--scale_factor', type=float, required=True)
parser.add_argument('--disable_batch_norm', dest='use_batch_norm',
                    action='store_false')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--pickle_dir', type=str, required=True)
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

    base_data_dir = {
        '/raid': '/raid/poggio/home/larend/data',
        '/om': '/om/user/larend/data',
        '/cbcl': '/cbcl/cbcl01/larend/data'}[FLAGS.host_filesystem]

    data_dir = {
        'cifar10': '{}/cifar-10-tfrecords'.format(base_data_dir),
        'imagenet': '{}/imagenet-tfrecords'.format(base_data_dir)}[FLAGS.dataset]

    # Get unperturbed predictions.
    model = estimator.Estimator(
        model_dir=FLAGS.model_dir,
        params={
            'batch_size': 100,
            'dataset': FLAGS.dataset,
            'use_batch_norm': FLAGS.use_batch_norm,
            'num_filters': num_filters},
        tf_random_seed=int(time.time()))

    imagenet_train_predict_shuffle_seed = int(time.time())

    unperturbed_predictions = []
    splits = ['train', 'validation']
    for split in splits:
        unperturbed_predictions.append(model.predict(
            data_dir=data_dir,
            split=split,
            imagenet_train_predict_shuffle_seed=imagenet_train_predict_shuffle_seed))

    for crossval in range(2):
        results = [None for _ in range(5)]

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

                for j, split in enumerate(splits):
                    # Build kernel file name.
                    split_str = {'train': '', 'validation': '_test'}[split]
                    kernel_filename = os.path.join(FLAGS.pickle_dir,
                                                   'kernel{}{}.pkl'.format(split_str, crossval))

                    t_0 = time.time()
                    same = model.robustness(
                        perturbation_type,
                        perturbation_amount,
                        kernel_filename,
                        unperturbed_predictions[j],
                        data_dir=data_dir,
                        split=split,
                        imagenet_train_predict_shuffle_seed=imagenet_train_predict_shuffle_seed)
                    t_1 = time.time()

                    print('crossval: {}'.format(crossval))
                    print('perturbation: {}'.format(perturbation_type))
                    print('amount: {}'.format(perturbation_amount))
                    print('split: {}'.format(split))
                    print('same: {}'.format(same))
                    print('time: {}'.format(t_1 - t_0))
                    sys.stdout.flush()

                    results[results_index][j][i] = same

        if not os.path.exists(FLAGS.pickle_dir):
            os.makedirs(FLAGS.pickle_dir)

        with open(os.path.join(FLAGS.pickle_dir, 'robustness{}.pkl'.format(crossval)), 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
