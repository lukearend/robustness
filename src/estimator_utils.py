"""Various utilities called throughout other scripts."""

import math
import os

import scipy.io

import tensorflow as tf

import estimator_datasets

from tensorflow.python.platform import tf_logging as logging
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training import device_setter
from tensorflow.contrib.learn.python.learn import monitors as monitor_lib
from tensorflow.tensorboard.backend.event_processing import event_accumulator


# Utils for estimator_datasets.
def get_mean_imagenet_rgb():
    """Fetches the mean activation per pixel over the entire ImageNet
    dataset from mean_imagenet_rgb.mat file, which must be located in
    the path shown."""
    # Load the data from file.
    if os.path.exists('/raid/poggio/home/larend/robust/prep/mean_imagenet_rgb.mat'):
        data = scipy.io.loadmat('/raid/poggio/home/larend/robust/prep/mean_imagenet_rgb.mat')
    elif os.path.exists('/cbcl/cbcl01/larend/robust/prep/mean_imagenet_rgb.mat'):
        data = scipy.io.loadmat('/cbcl/cbcl01/larend/robust/prep/mean_imagenet_rgb.mat')
    else:
        data = scipy.io.loadmat('/om/user/larend/robust/prep/mean_imagenet_rgb.mat')
    mean_imagenet_rgb = data['mean_imagenet_rgb']

    # Convert to float32 Tensor and map onto range [0, 1].
    mean_imagenet_rgb = tf.cast(mean_imagenet_rgb, tf.float32) * (1. / 255.)

    return mean_imagenet_rgb


# Utils for estimator_graph.
def get_dataset(dataset, mode, input_path, params, predict_split='validation',
                imagenet_train_predict_shuffle_seed=None,
                imagenet_train_predict_partial=False):
    """Get a dataset based on name."""
    if dataset == 'cifar10':
        return estimator_datasets.Cifar10Dataset(mode,
                                              input_path,
                                              params,
                                              predict_split=predict_split)
    elif dataset == 'imagenet':
        return estimator_datasets.ImageNetDataset(mode,
                                               input_path,
                                               params,
                                               predict_split=predict_split,
                                               imagenet_train_predict_shuffle_seed=imagenet_train_predict_shuffle_seed,
                                               imagenet_train_predict_partial=imagenet_train_predict_partial)
    else:
        raise ValueError("Dataset '{}' not recognized.".format(dataset))

def get_num_examples_per_epoch(dataset, mode):
    """Get number of examples per epoch for some dataset and mode."""
    if dataset == 'cifar10':
        return estimator_datasets.Cifar10Dataset.num_examples_per_epoch(mode)
    elif dataset == 'imagenet':
        return estimator_datasets.ImageNetDataset.num_examples_per_epoch(mode)
    else:
        raise ValueError("Dataset '{}' not recognized.".format(dataset))


# Utils for estimator_fns.
def local_device_setter(ps_device_type='cpu', worker_device='/cpu:0',
                        ps_strategy=None):
    """Set local device."""
    ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

    if ps_strategy is None:
        ps_strategy = device_setter._RoundRobinStrategy(1)

    def _local_device_chooser(op):
        """Choose local device."""
        current_device = pydev.DeviceSpec.from_string(op.device or "")

        node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
        if node_def.op in ps_ops:
            ps_device_spec = pydev.DeviceSpec.from_string(
                '/{}:{}'.format(ps_device_type, ps_strategy(op)))

            ps_device_spec.merge_from(current_device)
            return ps_device_spec.to_string()
        worker_device_spec = pydev.DeviceSpec.from_string(
            worker_device or "")
        worker_device_spec.merge_from(current_device)
        return worker_device_spec.to_string()
    return _local_device_chooser

def in_top_k(predictions, targets, k):
    """My implementation of tf.nn.in_top_k() which breaks ties by
    choosing the lowest index.

    Args:
        predictions: a [batch_size, classes] Tensor of probabilities.
        targets: a [batch_size] Tensor of class labels.
        k: number of top elements to look at for computing precision.

    Returns:
        correct: a [batch_size] float Tensor where 1.0 is correct and
                 0.0 is incorrect (float makes it simpler to average).
    """
    _, top_k_indices = tf.nn.top_k(predictions, k)

    # Initialize a False [batch_size] bool Tensor.
    last_correct = tf.logical_and(False, tf.is_finite(tf.to_float(targets)))
    # Loop over the top k predictions, checking if any are correct.
    for i in range(k):
        p_i = tf.squeeze(tf.slice(top_k_indices, [0, i], [-1, 1]), axis=1)
        correct = tf.logical_or(last_correct, tf.equal(tf.to_int64(targets),
                                                       tf.to_int64(p_i)))
        last_correct = correct

    return tf.to_float(correct)


# Utils.
class ExamplesPerSecondHook(session_run_hook.SessionRunHook):
    """Hook to print out examples per second. Tracks total time and
    divides by the total number of steps to get average step time.
    batch_size is then used to determine the running average of
    examples per second. Also logs examples per second for the most
    recent interval.
    """
    def __init__(self, batch_size, every_n_steps=100, every_n_secs=None):
        """Initializer for ExamplesPerSecondHook.
            Args:
                batch_size: Total batch size used to calculate
                            examples/second from global time.
                every_n_steps: Log stats every n steps.
                every_n_secs: Log stats every n seconds.
        """
        if (every_n_steps is None) == (every_n_secs is None):
            raise ValueError("Exactly one of every_n_steps and every_n_secs"
                             "should be provided.")
        self._timer = basic_session_run_hooks.SecondOrStepTimer(
            every_steps=every_n_steps, every_secs=every_n_secs)

        self._global_step_tensor = None
        self._step_train_time = 0
        self._total_steps = 0
        self._batch_size = batch_size

    def begin(self):
        self._global_step_tensor = training_util.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use StepCounterHook.")

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return basic_session_run_hooks.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        _ = run_context

        global_step = run_values.results
        if self._timer.should_trigger_for_step(global_step):
            (elapsed_time,
             elapsed_steps) = self._timer.update_last_triggered_step(
                                  global_step)
            if elapsed_time is not None:
                steps_per_sec = elapsed_steps / elapsed_time
                self._step_train_time += elapsed_time
                self._total_steps += elapsed_steps

                average_examples_per_sec = (self._batch_size *
                                            self._total_steps /
                                            self._step_train_time)
                current_examples_per_sec = steps_per_sec * self._batch_size
                logging.info(
                    "Average examples/sec: {:.2f} ({:.2f}), step = {}".format(
                        average_examples_per_sec,
                        current_examples_per_sec,
                        self._total_steps))


def replace_monitors_with_hooks(monitors_and_hooks, model):
    """Converts a list which may contain hooks or monitors to a list of
    hooks."""
    return monitor_lib.replace_monitors_with_hooks(monitors_and_hooks, model)
