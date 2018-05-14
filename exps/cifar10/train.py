"""Train a model."""

import os
import sys

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_index', type=int, required=True)
parser.add_argument('--host_filesystem', type=str, required=True)
FLAGS = parser.parse_args()

append_path = {
    '/raid': '/raid/poggio/home/larend/robust/src',
    '/om': '/om/user/larend/robust/src',
    '/cbcl': '/cbcl/cbcl01/larend/robust/src'}[FLAGS.host_filesystem]
sys.path.append(append_path)
import estimator


def main():
    name = str(FLAGS.model_index).zfill(5)
    num_filters, use_batch_norm = {0: (4, True),
                                   1: (8, True),
                                   2: (16, True),
                                   3: (32, True),
                                   4: (64, True),
                                   5: (4, False),
                                   6: (8, False),
                                   7: (16, False),
                                   8: (32, False),
                                   9: (64, False)}[FLAGS.model_index]

    base_model_dir, base_data_dir = {
        '/raid': ('/raid/poggio/home/larend/models',
                  '/raid/poggio/home/larend/data'),
        '/om': ('/om/user/larend/models',
                '/om/user/larend/data'),
        '/cbcl': ('/cbcl/cbcl01/larend/models',
                  '/cbcl/cbcl01/larend/data')}[FLAGS.host_filesystem]
    import estimator

    model = estimator.Estimator(
        model_dir='{}/robust/cifar10/{}'.format(base_model_dir, name),
        params={
            'initial_learning_rate': 0.1,
            'learning_rate_decay_factor': 0.1,
            'epochs_to_decay': [90, 135],
            'max_epochs': 180,
            'train_with_distortion': True,
            'momentum': 0.9,
            'batch_size': 128,
            'weight_decay': 0.0001,
            'dataset': 'cifar10',
            'use_batch_norm': use_batch_norm,
            'num_filters': num_filters},
        tf_random_seed=12345)

    model.train(data_dir='{}/cifar-10-tfrecords'.format(base_data_dir),
                num_gpus=2,
                save_summary_steps=1000,
                save_checkpoint_and_validate_secs=120,
                keep_checkpoint_max=10,
                keep_checkpoint_every_n_hours=1,
                early_stopping_epochs=None,
                log_every_n_steps=100)

    model.evaluate(data_dir='{}/cifar-10-tfrecords'.format(base_data_dir),
                   num_gpus=1)


if __name__ == '__main__':
    main()
