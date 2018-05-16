import numpy as np


class Dataset(object):

    def __init__(self):

        # # #
        # Dataset general
        self.dataset_path = ""
        self.proportion_training_set = 0.95
        self.shuffle_data = True

        # # #
        # For reusing tfrecords:
        self.reuse_TFrecords = False
        self.reuse_TFrecords_ID = 0
        self.reuse_TFrecords_path = ""

        # # #
        # Set random labels
        self.random_labels = False
        self.scramble_data = False

        # # #
        # Transfer learning
        self.transfer_learning = False
        self.transfer_pretrain = True
        self.transfer_label_offset = 0
        self.transfer_restart_name = "_pretrain"
        self.transfer_append_name = ""

        # Find dataset path:
        for line in open("data/paths", 'r'):
            if 'Dataset:' in line:
                self.dataset_path = line.split(" ")[1].replace('\r', '').replace('\n', '')

    # # #
    # Dataset general
    # Set base tfrecords
    def generate_base_tfrecords(self):
        self.reuse_TFrecords = False

    # Set reuse tfrecords mode
    def reuse_tfrecords(self, experiment):
        self.reuse_TFrecords = True
        self.reuse_TFrecords_ID = experiment.ID
        self.reuse_TFrecords_path = experiment.name

    # # #
    # Transfer learning
    def do_pretrain_transfer_lerning(self):
        self.transfer_learning = True
        self.transfer_append_name = self.transfer_restart_name

    def do_transfer_transfer_lerning(self):
        self.transfer_learning = True
        self.transfer_pretrain = True
        self.transfer_label_offset = 5
        self.transfer_append_name = "_transfer"


class DNN(object):

    def __init__(self):
        self.name = "Alexnet"
        self.pretrained = False
        self.version = 1
        self.layers = 4
        self.neuron_multiplier = np.ones([self.layers])

    def set_num_layers(self, num_layers):
        self.layers = num_layers
        self.neuron_multiplier = np.ones([self.layers])


class Hyperparameters(object):

    def __init__(self):
        self.batch_size = 128
        self.learning_rate = 1e-2
        self.num_epochs_per_decay = 1.0
        self.learning_rate_factor_per_decay = 0.95
        self.weight_decay = 0
        self.max_num_epochs = 60
        self.crop_size = 28
        self.image_size = 32
        self.drop_train = 1
        self.drop_test = 1
        self.momentum = 0.9
        self.augmentation = False


class Experiments(object):

    def __init__(self, id, name):
        self.name = "base"
        self.log_dir_base = "/om/user/xboix/share/robustness/models_final/"
            #"/om/user/xboix/src/robustness/robustness/log/"
            #"/Users/xboix/src/martin/robustness/robustness/log/"
            #"/om/user/xboix/src/robustness/robustness/log/"


        # Recordings
        self.max_to_keep_checkpoints = 5
        self.recordings = False
        self.num_batches_recordings = 0

        # Plotting
        self.plot_details = 0
        self.plotting = False

        # Test after training:
        self.test = False

        # Start from scratch even if it existed
        self.restart = False

        # Skip running experiments
        self.skip = False

        # Save extense summary
        self.extense_summary = True

        # Add ID to name:
        self.ID = id
        self.name = 'ID' + str(self.ID) + "_" + name

        # Add additional descriptors to Experiments
        self.dataset = Dataset()
        self.dnn = DNN()
        self.hyper = Hyperparameters()

    def do_recordings(self, max_epochs):
        self.max_to_keep_checkpoints = 0
        self.recordings = True
        self.hyper.max_num_epochs = max_epochs
        self.num_batches_recordings = 10

    def do_plotting(self, plot_details=0):
        self.plot_details = plot_details
        self.plotting = True

# # #
# Create set of experiments
opt = []
plot_freezing = []

neuron_multiplier = [0.25, 0.5, 1, 2, 4]
training_data = [1] #, 0.5, 0.25, 0.125, 0.0625]
name = ["Alexnet"]
num_layers = [5]
max_epochs = [200]


# 0-1, 32-33, 64-65
# 2-31, 34-63, 66-95

idx = 0
for d in training_data:
    # Create base for TF records:
    opt += [Experiments(idx, "data_" + str(d))]
    opt[-1].dataset.proportion_training_set *= d
    opt[-1].hyper.max_num_epochs = 0
    idx += 1

idx_base_random = idx
# Create base for TF records:
for d in training_data:
    # Create base for TF records:
    opt += [Experiments(idx, "data_random")]
    opt[-1].hyper.max_num_epochs = 0
    opt[-1].dataset.proportion_training_set *= d
    opt[-1].dataset.random_labels = True
    idx += 1


for name_NN, num_layers_NN, max_epochs_NN in zip(name, num_layers, max_epochs):
    for idx_data in range(len(training_data)):
        for flag_random in range(2):
            for regularizers in range(5):
                if flag_random and regularizers > 0:
                    continue

                # Change number neurons for each layer
                for multiplier in neuron_multiplier:
                    opt += [Experiments(idx, name_NN + "_layers_all_" +
                                    str(multiplier) + "_" + str(regularizers) + "_" + str(idx_data))]

                    opt[-1].hyper.max_num_epochs = max_epochs_NN
                    opt[-1].dnn.name = name_NN
                    opt[-1].dnn.set_num_layers(num_layers_NN)
                    opt[-1].dnn.neuron_multiplier.fill(multiplier)

                    if flag_random == 0:
                        opt[-1].dataset.reuse_tfrecords(opt[idx_data])
                        opt[-1].dataset.proportion_training_set *= training_data[idx_data]
                        opt[-1].hyper.max_num_epochs = int(max_epochs_NN / training_data[idx_data])
                        opt[-1].hyper.num_epochs_per_decay = \
                            int(opt[-1].hyper.num_epochs_per_decay / training_data[idx_data])

                    elif flag_random == 1:
                        opt[-1].dataset.reuse_tfrecords(opt[idx_data+idx_base_random])
                        opt[-1].dataset.random_labels = True
                        opt[-1].dataset.proportion_training_set *= training_data[idx_data]
                        opt[-1].hyper.max_num_epochs = int(max_epochs_NN * 10 / training_data[idx_data])
                        opt[-1].hyper.num_epochs_per_decay = int(opt[-1].hyper.num_epochs_per_decay  / training_data[idx_data])


                    # SKIP#SKIP#SKIP#SKIP#SKIP#SKIP
                    #if regularizers > 0:
                    #    opt[-1].skip = True

                    if regularizers == 1:
                        opt[-1].hyper.augmentation = True
                        opt[-1].hyper.max_num_epochs *= int(2)
                    elif regularizers == 2:
                        opt[-1].hyper.drop_train = 0.5
                    elif regularizers == 3:
                        opt[-1].hyper.weight_decay = 0.001
                    elif regularizers == 4:
                        opt[-1].hyper.augmentation = True
                        opt[-1].hyper.max_num_epochs *= int(2)
                        opt[-1].hyper.drop_train = 0.5
                        opt[-1].hyper.weight_decay = 0.001
                    idx += 1

training_data = [0.0625]

for d in training_data:
    # Create base for TF records:
    opt += [Experiments(idx, "data_" + str(d))]
    opt[-1].dataset.proportion_training_set *= d
    opt[-1].hyper.max_num_epochs = 0
    idx += 1

idx_base_random = idx
# Create base for TF records:
for d in training_data:
    # Create base for TF records:
    opt += [Experiments(idx, "data_random")]
    opt[-1].hyper.max_num_epochs = 0
    opt[-1].dataset.proportion_training_set *= d
    opt[-1].dataset.random_labels = True
    idx += 1


for name_NN, num_layers_NN, max_epochs_NN in zip(name, num_layers, max_epochs):
    for idx_data in range(len(training_data)):
        for flag_random in range(2):
            for regularizers in range(5):
                if flag_random and regularizers > 0:
                    continue

                # Change number neurons for each layer
                for multiplier in neuron_multiplier:
                    opt += [Experiments(idx, name_NN + "_layers_all_" +
                                    str(multiplier) + "_" + str(regularizers) + "_" + str(idx_data))]

                    opt[-1].hyper.max_num_epochs = max_epochs_NN
                    opt[-1].dnn.name = name_NN
                    opt[-1].dnn.set_num_layers(num_layers_NN)
                    opt[-1].dnn.neuron_multiplier.fill(multiplier)

                    if flag_random == 0:
                        opt[-1].dataset.reuse_tfrecords(opt[idx_data])
                        opt[-1].dataset.proportion_training_set *= training_data[idx_data]
                        opt[-1].hyper.max_num_epochs = int(max_epochs_NN / training_data[idx_data])
                        opt[-1].hyper.num_epochs_per_decay = \
                            int(opt[-1].hyper.num_epochs_per_decay / training_data[idx_data])

                    elif flag_random == 1:
                        opt[-1].dataset.reuse_tfrecords(opt[idx_data+idx_base_random])
                        opt[-1].dataset.random_labels = True
                        opt[-1].dataset.proportion_training_set *= training_data[idx_data]
                        opt[-1].hyper.max_num_epochs = int(max_epochs_NN * 10 / training_data[idx_data])
                        opt[-1].hyper.num_epochs_per_decay = int(opt[-1].hyper.num_epochs_per_decay  / training_data[idx_data])


                    # SKIP#SKIP#SKIP#SKIP#SKIP#SKIP
                    #if regularizers > 0:
                    #    opt[-1].skip = True

                    if regularizers == 1:
                        opt[-1].hyper.augmentation = True
                        opt[-1].hyper.max_num_epochs *= int(2)
                    elif regularizers == 2:
                        opt[-1].hyper.drop_train = 0.5
                    elif regularizers == 3:
                        opt[-1].hyper.weight_decay = 0.001
                    elif regularizers == 4:
                        opt[-1].hyper.augmentation = True
                        opt[-1].hyper.max_num_epochs *= int(2)
                        opt[-1].hyper.drop_train = 0.5
                        opt[-1].hyper.weight_decay = 0.001
                    idx += 1



training_data = [0.015625]

for d in training_data:
    # Create base for TF records:
    opt += [Experiments(idx, "data_" + str(d))]
    opt[-1].dataset.proportion_training_set *= d
    opt[-1].hyper.max_num_epochs = 0
    idx += 1

idx_base_random = idx
# Create base for TF records:
for d in training_data:
    # Create base for TF records:
    opt += [Experiments(idx, "data_random")]
    opt[-1].hyper.max_num_epochs = 0
    opt[-1].dataset.proportion_training_set *= d
    opt[-1].dataset.random_labels = True
    idx += 1


for name_NN, num_layers_NN, max_epochs_NN in zip(name, num_layers, max_epochs):
    for idx_data in range(len(training_data)):
        for flag_random in range(2):
            for regularizers in range(5):
                if flag_random and regularizers > 0:
                    continue

                # Change number neurons for each layer
                for multiplier in neuron_multiplier:
                    opt += [Experiments(idx, name_NN + "_layers_all_" +
                                    str(multiplier) + "_" + str(regularizers) + "_" + str(idx_data))]

                    ###ASDAAD
                    #ASDASDAS
                    opt[-1].restart = True
                    opt[-1].hyper.learning_rate = 1e-1

                    opt[-1].hyper.max_num_epochs = max_epochs_NN
                    opt[-1].dnn.name = name_NN
                    opt[-1].dnn.set_num_layers(num_layers_NN)
                    opt[-1].dnn.neuron_multiplier.fill(multiplier)

                    if flag_random == 0:
                        opt[-1].dataset.reuse_tfrecords(opt[idx_data])
                        opt[-1].dataset.proportion_training_set *= training_data[idx_data]
                        opt[-1].hyper.max_num_epochs = int(max_epochs_NN / training_data[idx_data])
                        opt[-1].hyper.num_epochs_per_decay = \
                            int(opt[-1].hyper.num_epochs_per_decay / training_data[idx_data])

                    elif flag_random == 1:
                        opt[-1].dataset.reuse_tfrecords(opt[idx_data+idx_base_random])
                        opt[-1].dataset.random_labels = True
                        opt[-1].dataset.proportion_training_set *= training_data[idx_data]
                        opt[-1].hyper.max_num_epochs = int(max_epochs_NN * 10 / training_data[idx_data])
                        opt[-1].hyper.num_epochs_per_decay = int(opt[-1].hyper.num_epochs_per_decay  / training_data[idx_data])


                    # SKIP#SKIP#SKIP#SKIP#SKIP#SKIP
                    #if regularizers > 0:
                    #    opt[-1].skip = True

                    if regularizers == 1:
                        opt[-1].hyper.augmentation = True
                        opt[-1].hyper.max_num_epochs *= int(2)
                    elif regularizers == 2:
                        opt[-1].hyper.drop_train = 0.5
                    elif regularizers == 3:
                        opt[-1].hyper.weight_decay = 0.001
                    elif regularizers == 4:
                        opt[-1].hyper.augmentation = True
                        opt[-1].hyper.max_num_epochs *= int(2)
                        opt[-1].hyper.drop_train = 0.5
                        opt[-1].hyper.weight_decay = 0.001
                    idx += 1
