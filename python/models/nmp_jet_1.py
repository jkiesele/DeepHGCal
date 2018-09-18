import tensorflow as tf
from models.sparse_conv import SparseConv
from ops.sparse_conv import *
from ops.mpn_jet import message_parsing_network


class NeuralMessagePassingJet1(SparseConv):

    def __init__(self, n_space, n_space_local, n_all, n_max_neighbors, batch_size, max_entries, num_classes,
                 learning_rate=0.0001):
        super(NeuralMessagePassingJet1, self).__init__(n_space, n_space_local, n_all, n_max_neighbors, batch_size, max_entries,
                                          num_classes, learning_rate)

    def _find_logits(self):
        input = self._placeholder_all_features
        input = tf.concat((self._placeholder_all_features, self._placeholder_space_features_local, self._placeholder_space_features), axis=2)

        net = tf.layers.dense(input, units=10, activation=tf.nn.relu)

        logits = message_parsing_network(input, net, 3, self.num_classes)
        self._graph_temp = logits

        return logits

    def get_variable_scope(self):
        return 'neural_message_parsing_network_1'
