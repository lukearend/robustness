"""Train a model."""

import os
import sys
import time
import argparse

append_path = {
    '/raid': '/raid/poggio/home/larend/robust/src',
    '/om': '/om/user/larend/robust/src',
    '/cbcl': '/cbcl/cbcl01/larend/robust/src'}[FLAGS.host_filesystem]
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
        results = []

        model = Estimator(
            model_dir=FLAGS.model_dir,
            params={
                'batch_size': 100,
                'dataset': FLAGS.dataset,
                'use_batch_norm': FLAGS.use_batch_norm,
                'num_filters': num_filters},
            tf_random_seed=int(time.time()))

        for k, perturbation_type in enumerate([0]):
            perturbation_amounts = {
                0: np.linspace(0.0, 1.0, 7),
                1: np.linspace(0.0, 1.0, 7),
                2: np.linspace(0.0, 1.0, 7)}[perturbation_type]

            results.append([np.zeros(len(perturbation_amounts)) for _ in range(2)])
            for i, perturbation_amount in enumerate(perturbation_amounts):

                for j, split in enumerate(['validation', 'train']):
                    accuracy = model.robustness(
                        perturbation_type,
                        perturbation_amount,
                        kill_mask,
                        data_dir=data_dir,
                        split=split)

                    results[k][j][i] = accuracy

        if not os.path.exists(FLAGS.out_dir):
            os.makedirs(FLAGS.out_dir)

        with open(os.path.join(FLAGS.out_dir, 'robustness{}.pkl'.format(crossval)), 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

        tf.reset_default_graph()


if __name__ == '__main__':
    main()
