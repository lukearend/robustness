"""The model_fn and input_fn."""

import itertools
import math

import tensorflow as tf

import estimator_utils
import estimator_datasets
import estimator_graph


def input_fn_read_labels(mode, input_path, params, num_gpus=None,
             predict_split='validation', imagenet_train_predict_shuffle_seed=None,
             imagenet_train_predict_partial=False):
    """Create input graph for model.

    Args:
        mode: one of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}.
        input_path: directory where input data is located.
        params: hyperparameters for training and architecture.
        num_gpus: number of GPUs participating in data-parallelism.

    Returns:
        feature_shards: list of length num_gpus containing feature
                        batches.
        label_shards: list of length num_gpus containing label batches.
    """
    with tf.device('/cpu:0'):
        if mode == tf.estimator.ModeKeys.PREDICT:
            dataset = estimator_utils.get_dataset(params['dataset'],
                                                  mode,
                                                  input_path,
                                                  params,
                                                  predict_split=predict_split,
                                                  imagenet_train_predict_shuffle_seed=imagenet_train_predict_shuffle_seed,
                                                  imagenet_train_predict_partial=imagenet_train_predict_partial)
            _, label_batch = dataset.make_batch(params['batch_size'])
            return label_batch


def input_fn(mode, input_path, params, num_gpus=None,
             predict_split='validation', imagenet_train_predict_shuffle_seed=None,
             imagenet_train_predict_partial=False):
    """Create input graph for model.

    Args:
        mode: one of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}.
        input_path: directory where input data is located.
        params: hyperparameters for training and architecture.
        num_gpus: number of GPUs participating in data-parallelism.

    Returns:
        feature_shards: list of length num_gpus containing feature
                        batches.
        label_shards: list of length num_gpus containing label batches.
    """
    with tf.device('/cpu:0'):
        if mode == tf.estimator.ModeKeys.PREDICT:
            dataset = estimator_utils.get_dataset(params['dataset'],
                                                  mode,
                                                  input_path,
                                                  params,
                                                  predict_split=predict_split,
                                                  imagenet_train_predict_shuffle_seed=imagenet_train_predict_shuffle_seed,
                                                  imagenet_train_predict_partial=imagenet_train_predict_partial)
            image_batch, _ = dataset.make_batch(params['batch_size'])
            return {'images': image_batch}, None

        elif mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
            # Set the batch size.
            batch_size = params['batch_size']

            if mode == tf.estimator.ModeKeys.EVAL:
                # The validation batch size must evenly divide the
                # validation set to avoid the risk of getting a batch
                # with fewer than [the number of GPU] examples. Here we
                # find the largest factor of the validation set size
                # which is smaller than params['batch_size'] to use as
                # the batch size during validation. Avoid preparing a
                # validation data set with a prime number of examples.
                num_validation_examples = (
                    estimator_utils.get_num_examples_per_epoch(
                        params['dataset'], tf.estimator.ModeKeys.EVAL))
                while True:
                    if num_validation_examples % batch_size == 0:
                        break
                    batch_size = batch_size - 1

            dataset = estimator_utils.get_dataset(params['dataset'],
                                               mode,
                                               input_path,
                                               params)

            image_batch, label_batch = dataset.make_batch(batch_size)

        if num_gpus <= 1:
            return [image_batch], [label_batch]

        # Shard-up the batch among GPUs. Assumes image_batch will always
        # have batch_size elements since batch_size either divides the
        # data set size perfectly or the data repeats indefinitely.
        image_batch = tf.unstack(image_batch, num=batch_size, axis=0)
        label_batch = tf.unstack(label_batch, num=batch_size, axis=0)
        feature_shards = [[] for i in range(num_gpus)]
        label_shards = [[] for i in range(num_gpus)]

        for i in range(batch_size):
            idx = i % num_gpus
            feature_shards[idx].append(image_batch[i])
            label_shards[idx].append(label_batch[i])
        feature_shards = [tf.stack(x) for x in feature_shards]
        label_shards = [tf.stack(x) for x in label_shards]

        return feature_shards, label_shards


def get_model_fn(num_gpus, variable_strategy='GPU', keep_checkpoint_max=10,
                 keep_checkpoint_every_n_hours=2,
                 test_robustness=False,
                 perturbation_type=None,
                 perturbation_amount=None,
                 kill_mask=None):
    """Returns a function that will build the estimator.

    Args:
        num_gpus: number of GPUs to use for training.
        variable_strategy: one of {'CPU', 'GPU'}.
                           If 'CPU', CPU is the parameter server and
                           manages gradient updates.
                           If 'GPU', parameters are distributed evenly
                           across all GPUs and the first GPU manages
                           gradient updates. This is optimal with
                           interconnected K80s.
        keep_checkpoint_max: number of recent checkpoint to keep during
                             training.
        keep_checkpoint_every_n_hours: how frequently to keep a
                                       checkpoint permanently.
        test_robustness: whether to build and use the graph for testing
                         robustness.
        perturbation_type, perturbation_amount, kill_mask:
            settings for testing robustness.

    Returns:
        The estimator model_fn.
    """

    def _estimator_model_fn(features, labels, mode, params):
        """Model body.

        Args:
            features: list of Tensors, one for each tower. If mode is
                      PREDICT, Tensor.
            labels: list of Tensors, one for each tower. If mode is
                    PREDICT, None.
            mode: one of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}.
            params: hyperparameters for training and architecture.
        Returns:
            EstimatorSpec object.
        """
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        if mode == tf.estimator.ModeKeys.PREDICT:
            if test_robustness:
                logits = estimator_graph.forward_pass_test(features,
                                                           params,
                                                           perturbation_type,
                                                           perturbation_amount,
                                                           kill_mask)

                predictions = {
                    'classes': tf.argmax(logits, axis=1)
                }

            else:
                logits, activations = estimator_graph.forward_pass(features,
                                                      is_training,
                                                      params)

                predictions = {
                    'classes': tf.argmax(logits, axis=1),
                    'activations': activations
                }

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions)

        elif mode in [tf.estimator.ModeKeys.TRAIN,
                      tf.estimator.ModeKeys.EVAL]:
            tower_features = features
            tower_labels = labels
            tower_losses = []
            tower_gradvars = []
            tower_predictions = []
            weight_loss = None

            if num_gpus == 0:
                num_devices = 1
                device_type = 'cpu'
            else:
                num_devices = num_gpus
                device_type = 'gpu'

            for i in range(num_devices):
                worker_device = '/{}:{}'.format(device_type, i)
                if variable_strategy == 'CPU':
                    device_setter = estimator_utils.local_device_setter(
                        worker_device=worker_device)
                elif variable_strategy == 'GPU':
                    device_setter = estimator_utils.local_device_setter(
                        ps_device_type='gpu',
                        worker_device=worker_device,
                        ps_strategy=\
                            tf.contrib.training.GreedyLoadBalancingStrategy(
                                num_gpus,
                                tf.contrib.training.byte_size_load_fn))

                # Compute loss and gradients for each tower.
                with tf.variable_scope('estimator', reuse=bool(i != 0)):
                    with tf.name_scope('tower_{}'.format(i)):
                        with tf.device(device_setter):
                            (loss,
                             gradvars,
                             predictions,
                             weight_loss) = _tower_fn(tower_features[i],
                                                      tower_labels[i],
                                                      is_training,
                                                      params)

                            tower_losses.append(loss)
                            tower_gradvars.append(gradvars)
                            tower_predictions.append(predictions)

            # Compute global loss and gradients.
            gradvars = []
            with tf.name_scope('gradient_averaging'):
                all_grads = {}
                for grad, var in itertools.chain(*tower_gradvars):
                    if grad is not None:
                        all_grads.setdefault(var, []).append(grad)
                for var, grads in all_grads.items():
                    # Average gradients on device holding their
                    # variables.
                    with tf.device(var.device):
                        if len(grads) == 1:
                            avg_grads = grads[0]
                        else:
                            avg_grads = tf.multiply(tf.add_n(grads),
                                                    1. / len(grads))
                    gradvars.append((avg_grads, var))

            # Set device that runs ops to apply global gradient updates.
            if variable_strategy == 'GPU':
                consolidation_device = '/gpu:0'
            else:
                consolidation_device = '/cpu:0'
            with tf.device(consolidation_device):
                # Calculate the learning rate decay schedule.
                global_step = tf.train.get_or_create_global_step()
                num_examples_per_epoch = (
                    estimator_utils.get_num_examples_per_epoch(
                        params['dataset'], tf.estimator.ModeKeys.TRAIN))

                if params['dataset'] == 'cifar10' and 'epochs_to_decay' in params:
                    step_decay_boundaries = [
                        int(num_examples_per_epoch /
                            params['batch_size'] *
                            epoch_decay_boundary)
                        for epoch_decay_boundary in params['epochs_to_decay']]

                    values = [
                        params['initial_learning_rate'] * params['learning_rate_decay_factor'] ** i
                        for i in range(len(step_decay_boundaries) + 1)]

                    learning_rate = tf.train.piecewise_constant(
                        tf.to_int32(global_step),   # Must be int32 here.
                        step_decay_boundaries,
                        values)
                else:
                    num_steps_per_decay = int(num_examples_per_epoch /
                                              params['batch_size'] *
                                              params['num_epochs_per_decay'])

                    learning_rate = tf.train.exponential_decay(
                        tf.to_float(params['initial_learning_rate']),
                        global_step,
                        num_steps_per_decay,
                        params['learning_rate_decay_factor'],
                        staircase=True,
                        name='learning_rate')

                # Create momentum gradient descent optimizer.
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate,
                    params['momentum'])

                # Create the train op.
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.apply_gradients(gradvars,
                                                         global_step=global_step)

                # Combine results and compute metrics; name tensors we
                # want to log.
                loss = tf.reduce_mean(tower_losses, name='loss')
                weight_loss = tf.identity(weight_loss, name='weight_loss')
                cross_entropy_loss = tf.subtract(loss,
                                                 weight_loss,
                                                 name='cross_entropy_loss')

                predictions = {
                    'classes': tf.concat(
                        [p['classes'] for p in tower_predictions],
                        axis=0),
                    'probabilities': tf.concat(
                        [p['probabilities'] for p in tower_predictions],
                        axis=0)}
                stacked_labels = tf.concat(labels, axis=0)

                in_top_one = estimator_utils.in_top_k(
                    predictions['probabilities'],
                    stacked_labels,
                    1)
                in_top_five = estimator_utils.in_top_k(
                    predictions['probabilities'],
                    stacked_labels,
                    5)

                precision_at_one = tf.reduce_mean(in_top_one,
                                                  name='precision_at_one')
                precision_at_five = tf.reduce_mean(in_top_five,
                                                   name='precision_at_five')

                eval_metric_ops = {
                    'loss/loss': tf.metrics.mean(loss),
                    'precision/precision_at_one':
                        tf.metrics.mean(in_top_one),
                    'precision/precision_at_five':
                        tf.metrics.mean(in_top_five)}

            # Write summaries to monitor the training process.
            if is_training:
                # Plot helpful training metrics.
                tf.summary.scalar('global_step/learning_rate',
                                  learning_rate)
                tf.summary.scalar('loss/loss', loss)
                tf.summary.scalar('loss/weight_loss', weight_loss)
                tf.summary.scalar('loss/cross_entropy_loss',
                                  cross_entropy_loss)
                tf.summary.scalar('precision/precision_at_one',
                                  precision_at_one)
                tf.summary.scalar('precision/precision_at_five',
                                  precision_at_five)

                # Visualize a few training examples.
                tf.summary.image('examples/images',
                                 tower_features[0])

            # Merge the summaries for saving.
            summary_op = tf.summary.merge_all()

            # Create a saver for saving to checkpoints.
            saver = tf.train.Saver(
                max_to_keep=keep_checkpoint_max,
                keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

            scaffold = tf.train.Scaffold(summary_op=summary_op,
                                         saver=saver)

            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions,
                                              loss=loss,
                                              train_op=train_op,
                                              eval_metric_ops=eval_metric_ops,
                                              scaffold=scaffold)

    return _estimator_model_fn

def _tower_fn(features, labels, is_training, params):
    """Build computation tower.

    Args:
        features: a batch of images.
        labels: a batch of labels.
        is_training: bool, true if training graph.
        params: hyperparameters specifying architecture of the model.

    Returns:
        loss: loss for the tower.
        gradvars: a tuple containing the gradients and parameters.
        predictions: unscaled logit predictions.
        weight_loss: total loss from L2 penalty on model weights.
    """
    logits, _ = estimator_graph.forward_pass(features,
                                          is_training,
                                          params)

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits)
    }

    cross_entropy_loss = tf.reduce_mean(
        tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits))
    model_params = tf.trainable_variables()
    weight_loss = (params['weight_decay'] *
                   tf.add_n([tf.nn.l2_loss(v) for v in model_params]))
    loss = cross_entropy_loss + weight_loss

    grads = tf.gradients(loss, model_params)
    gradvars = zip(grads, model_params)

    return loss, gradvars, predictions, weight_loss
