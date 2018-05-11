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
    num_filters = {0: 16,
                   1: 32,
                   2: 64,
                   3: 128,
                   4: 256}[FLAGS.model_index]

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
            'num_epochs_per_decay': 30,
            'max_epochs': 120,
            'train_with_distortion': True,
            'momentum': 0.9,
            'batch_size': 256,
            'weight_decay': 0.0001,
            'dataset': 'cifar10',
            'num_filters': num_filters},
        tf_random_seed=12345)

    # model.train(data_dir='{}/cifar-10-tfrecords'.format(base_data_dir),
    #             num_gpus=8,
    #             save_summary_steps=100,
    #             save_checkpoint_and_validate_secs=180,
    #             keep_checkpoint_max=10,
    #             keep_checkpoint_every_n_hours=1,
    #             early_stopping_epochs=None,
    #             log_every_n_steps=100)

    model.evaluate(data_dir='{}/cifar-10-tfrecords'.format(base_data_dir),
                   num_gpus=1)


if __name__ == '__main__':
    main()
