"""Train a model."""

import os
import sys

import argparse

sys.path.append('/om/user/larend/robust/src')
sys.path.append('/raid/poggio/home/larend/robust/src')
import estimator


def main():
    name = str(FLAGS.model_index).zfill(5)
    num_filters = {0: 16,
                   1: 32,
                   2: 64,
                   3: 128,
                   4: 256}[FLAGS.model_index]

    if FLAGS.host_machine == 'dgx':
        base_model_dir = '/raid/poggio/home/larend/models'
        base_data_dir = '/raid/poggio/home/larend/data'
    else:
        base_model_dir = '/om/user/larend/models'
        base_data_dir = '/om/user/larend/data'

    model = estimator.Estimator(
        model_dir='{}/robust/imagenet/{}'.format(base_model_dir, name),
        params={
            'initial_learning_rate': 0.1,
            'learning_rate_decay_factor': 0.1,
            'num_epochs_per_decay': 30,
            'max_epochs': 120,
            'train_with_distortion': True,
            'momentum': 0.9,
            'batch_size': 256,
            'weight_decay': 0.0001,
            'dataset': 'imagenet',
            'num_filters': num_filters}
    tf_random_seed=12345)

    model.train(data_dir='{}/imagenet-tfrecords'.format(base_data_dir),
                num_gpus=8,
                save_summary_steps=10000,
                save_checkpoint_and_validate_secs=3600,
                keep_checkpoint_max=10,
                keep_checkpoint_every_n_hours=1,
                early_stopping_epochs=None,
                log_every_n_steps=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_index', type=int, required=True)
    parser.add_argument('--host_machine', type=str, default='om')
    FLAGS = parser.parse_args()

    main()
