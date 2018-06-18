import tensorflow as tf
from models.sparse_conv import SparseConv
from ops.sparse_conv import *


class SparseConv6(SparseConv):

    def __init__(self, n_space, n_space_local, n_all, n_max_neighbors, batch_size, max_entries, num_classes,
                 learning_rate=0.0001):
        super(SparseConv6, self).__init__(n_space, n_space_local, n_all, n_max_neighbors, batch_size, max_entries,
                                          num_classes, learning_rate)
        self.weight_weights = []


    def _find_logits(self):
        # # nl_all = tf.layers.dense(tf.scalar_mul(0.001, self._placeholder_all_features), units=8, activation=tf.nn.relu)
        # # nl_all = tf.layers.dense(nl_all, units=8, activation=tf.nn.relu)
        # # nl_all = tf.layers.dense(nl_all, units=8, activation=tf.nn.relu)
        #
        # # TODO: Remove it later after regenerating the data, this only picks energy (or do something similar)
        # net = self._placeholder_all_features
        # net = tf.concat((net, self._placeholder_space_features_local), axis=2)

        _input = construct_sparse_io_dict(self._placeholder_all_features, self._placeholder_space_features, self._placeholder_space_features_local,
                                          tf.squeeze(self._placeholder_num_entries))

        net = _input

        net = sparse_conv_2(net, num_neighbors=9, num_filters=9, n_prespace_conditions=5)
        net = sparse_conv_2(net, num_neighbors=9, num_filters=9, n_prespace_conditions=5)
        net = sparse_conv_2(net, num_neighbors=9, num_filters=9, n_prespace_conditions=5)
        net = sparse_max_pool_factored(net, 3)
        net = sparse_conv_2(net, num_neighbors=9, num_filters=9, n_prespace_conditions=5)
        net = sparse_max_pool_factored(net, 2)
        net = sparse_conv_2(net, num_neighbors=9, num_filters=9, n_prespace_conditions=5)
        net = sparse_max_pool_factored(net, 2)
        net = sparse_conv_2(net, num_neighbors=9, num_filters=9, n_prespace_conditions=5)
        net = sparse_max_pool_factored(net, 5)

        flattened_features = sparse_merge_flat(net, combine_three=False)
        self._graph_temp = flattened_features[:,0:10]

        fc_1 = tf.layers.dense(flattened_features, units=30, activation=tf.nn.relu)
        fc_2 = tf.layers.dense(fc_1, units=30, activation=tf.nn.relu)
        fc_3 = tf.layers.dense(fc_2, units=self.num_classes, activation=None)

        return fc_3

    def get_variable_scope(self):
        return 'sparse_conv_v4'
