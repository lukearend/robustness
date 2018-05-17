"""Train a model."""

import os
import sys
import time
import argparse
import pickle

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

    num_layers = {
        'cifar10': 19,
        'imagenet': 17}['cifar10']

    base_data_dir = {
        '/raid': '/raid/poggio/home/larend/data',
        '/om': '/om/user/larend/data',
        '/cbcl': '/cbcl/cbcl01/larend/data'}['/cbcl']

    data_dir = {
        'cifar10': '{}/cifar-10-tfrecords'.format(base_data_dir),
        'imagenet': '{}/imagenet-tfrecords'.format(base_data_dir)}['cifar10']

    for crossval in range(1):
        print('crossval: {}'.format(crossval))

        model = estimator.Estimator(
            model_dir='/cbcl/cbcl01/larend/models/robust/cifar10/00002',
            params={
                'batch_size': 100,
                'dataset': 'cifar10',
                'use_batch_norm': True,
                'num_filters': num_filters},
            tf_random_seed=int(time.time()))

        for split in ['validation', 'train']:
            print('split: {}'.format(split))

            t_0 = time.time()
            activations, labels, accuracy = model.activations(
                num_layers,
                data_dir=data_dir,
                split=split)
            t_1 = time.time()

            print('accuracy: {}'.format(accuracy))
            print('time: {}'.format(t_1 - t_0))

            out_dir = '/cbcl/cbcl01/larend/tmp'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            split_str = {'train': '', 'validation': '_test'}[split]

            with open(os.path.join(out_dir, 'activations{}{}.pkl'.format(split_str, crossval)), 'wb') as f:
                pickle.dump(activations, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(out_dir, 'labels{}{}.pkl'.format(split_str, crossval)), 'wb') as f:
                pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(out_dir, 'accuracy{}{}.pkl'.format(split_str, crossval)), 'wb') as f:
                pickle.dump(accuracy, f, protocol=pickle.HIGHEST_PROTOCOL)

        tf.reset_default_graph()

    print('done :)')


if __name__ == '__main__':
    main()
