"""Train a model."""

import os
import sys
import time
import argparse

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

    data_dir = {
        'cifar10': '/om/user/larend/data/cifar-10-tfrecords',
        'imagenet': '/om/user/larend/data/imagenet-tfrecords'}['cifar10']

    for crossval in range(1):
        model = Estimator(
            model_dir='/cbcl/cbcl01/larend/models/robust/cifar10/00002',
            params={
                'batch_size': 100,
                'dataset': 'cifar10',
                'use_batch_norm': True,
                'num_filters': num_filters},
            tf_random_seed=int(time.time()))

        for split in ['validation']:
            activations, labels, accuracy = model.activations(
                data_dir=data_dir,
                split=split)

            out_dir = '/cbcl/cbcl01/larend/tmp'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            split_str = {'train': '', 'validation': '_test'}[split]

            with open(os.path.join(FLAGS.out_dir, 'activations{}{}.pkl'.format(split_str, crossval)), 'wb') as f:
                pickle.dump(activations, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(FLAGS.out_dir, 'labels{}{}.pkl'.format(split_str, crossval)), 'wb') as f:
                pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(FLAGS.out_dir, 'accuracy{}{}.pkl'.format(split_str, crossval)), 'wb') as f:
                pickle.dump(accuracy, f, protocol=pickle.HIGHEST_PROTOCOL)

        tf.reset_default_graph()


if __name__ == '__main__':
    main()
