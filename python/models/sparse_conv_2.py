import tensorflow as tf
from models.sparse_conv import SparseConv
from ops.sparse_conv import *


class SparseConv2(SparseConv):

    def __init__(self, n_space, n_space_local, n_all, n_max_neighbors, batch_size, max_entries, num_classes,
                 learning_rate=0.0001):
        super(SparseConv2, self).__init__(n_space, n_space_local, n_all, n_max_neighbors, batch_size, max_entries,
                                          num_classes, learning_rate)

    def _find_logits(self):
        _input = construct_sparse_io_dict(tf.scalar_mul(0.001, self._placeholder_all_features),
                                          self._placeholder_space_features, self._placeholder_space_features_local,
                                          tf.squeeze(self._placeholder_num_entries))

        net = sparse_conv(_input, num_neighbors=10, output_all=15)
        net = sparse_conv(net, num_neighbors=10, output_all=30)
        net = sparse_conv(net, num_neighbors=10, output_all=45)
        net = sparse_max_pool(net, 1000)
        net = sparse_conv(net, num_neighbors=10, output_all=60)
        net = sparse_max_pool(net, 500)
        net = sparse_conv(net, num_neighbors=10, output_all=80)
        flattened_features = sparse_merge_flat(net, combine_three=True)

        fc_1 = tf.layers.dense(flattened_features, units=100, activation=tf.nn.relu,
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
        return 'sparse_conv_v2'
