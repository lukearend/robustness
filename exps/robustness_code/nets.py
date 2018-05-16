import tensorflow as tf

from models.mlp import MLP1 as net_MLP1
from models.mlp import MLP3 as net_MLP3
from models.alexnet import Alexnet as net_Alexnet
from models.alexnet import Alexnet_test as net_Alexnet_test
from utils import summary as summ


def MLP3(x, dropout_rate, opt, labels_id):
    return net_MLP3(x, opt, labels_id, dropout_rate)


def MLP1(x, dropout_rate, opt, labels_id):
    return net_MLP1(x, opt, labels_id, dropout_rate)


def Alexnet(x, dropout_rate, opt, labels_id):
    return net_Alexnet(x, opt, labels_id, dropout_rate)

def Alexnet_test(x, dropout_rate, select, opt, labels_id, robustness, robustness_graph):
    return net_Alexnet_test(x, opt, select, labels_id, dropout_rate, robustness, robustness_graph)
