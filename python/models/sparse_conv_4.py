import tensorflow as tf
from models.sparse_conv import SparseConv
from ops.sparse_conv import *


class SparseConv4(SparseConv):

    def __init__(self, n_space, n_space_local, n_all, n_max_neighbors, batch_size, max_entries, num_classes,
                 learning_rate=0.0001):
        super(SparseConv4, self).__init__(n_space, n_space_local, n_all, n_max_neighbors, batch_size, max_entries,
                                          num_classes, learning_rate)

    def _find_logits(self):
        nl_all = tf.layers.dense(tf.scalar_mul(0.001, self._placeholder_all_features), units=8, activation=tf.nn.relu)
        nl_all = tf.layers.dense(nl_all, units=8, activation=tf.nn.relu)
        nl_all = tf.layers.dense(nl_all, units=8, activation=tf.nn.relu)

        _input = construct_sparse_io_dict(nl_all, self._placeholder_space_features, self._placeholder_space_features_local,
                                          tf.squeeze(self._placeholder_num_entries))

        net = _input

        net = sparse_conv(net, num_neighbors=9, output_all=12)
        net = sparse_conv(net, num_neighbors=9, output_all=12)
        net = sparse_conv(net, num_neighbors=9, output_all=12)
        net = sparse_max_pool(net, 1000)
        net = sparse_conv(net, num_neighbors=9, output_all=12)
        net = sparse_max_pool(net, 500)
        net = sparse_conv(net, num_neighbors=9, output_all=12)
        net = sparse_max_pool(net, 250)
        net = sparse_conv(net, num_neighbors=9, output_all=12)
        net = sparse_max_pool(net, 50)

        flattened_features = sparse_merge_flat(net, combine_three=True)

        fc_1 = tf.layers.dense(flattened_features, units=30, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(mean=0., stddev=1),
                               bias_initializer=tf.zeros_initializer())
        fc_2 = tf.layers.dense(fc_1, units=30, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width),
                               bias_initializer=tf.zeros_initializer())
        fc_3 = tf.layers.dense(fc_2, units=self.num_classes, activation=None,
                               kernel_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width),
                               bias_initializer=tf.zeros_initializer())

        self._graph_temp = flattened_features

        return fc_3

    def get_variable_scope(self):
        return 'sparse_conv_v4'
