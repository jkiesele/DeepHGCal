import tensorflow as tf
from models.sparse_conv import SparseConv
from ops.sparse_conv import *


class SparseConv8(SparseConv):

    def __init__(self, n_space, n_space_local, n_all, n_max_neighbors, batch_size, max_entries, num_classes,
                 learning_rate=0.0001):
        super(SparseConv8, self).__init__(n_space, n_space_local, n_all, n_max_neighbors, batch_size, max_entries,
                                          num_classes, learning_rate)
        self.weight_weights = []


    def _find_logits(self):
        # # nl_all = tf.layers.dense(tf.scalar_mul(0.001, self._placeholder_all_features), units=8, activation=tf.nn.relu)
        # # nl_all = tf.layers.dense(nl_all, units=8, activation=tf.nn.relu)
        # # nl_all = tf.layers.dense(nl_all, units=8, activation=tf.nn.relu)
        #
        # # TODO: Remove it later after regenerating the data, this only picks energy (or do something similar)
        #net = self._placeholder_all_features
        net = self._placeholder_all_features[:, :, 3]/1000
        net = tf.expand_dims(net, axis=2)
        
        layers = self._placeholder_all_features[:, :, 4]
        layers = tf.expand_dims(layers, axis=2)
        space = self._placeholder_space_features[:,:,:2]
        
        space = tf.concat([space,layers], axis=-1)

        _input = construct_sparse_io_dict(net, space, self._placeholder_space_features_local,
                                          tf.squeeze(self._placeholder_num_entries))

        net = _input
        #net = sparse_max_pool(net, 2000)
        net = sparse_conv_loop(net, num_neighbors=100,  num_filters=32,  space_depth=2, space_relu=4, space_gauss=3)
        net = sparse_max_pool(net, 1)

        flattened_features = sparse_merge_flat(net, combine_three=False)
        self._graph_temp = flattened_features[:,0:10]

        fc_1 = tf.layers.dense(flattened_features, units=30, activation=tf.nn.relu)
        fc_2 = tf.layers.dense(fc_1, units=30, activation=tf.nn.relu)
        fc_3 = tf.layers.dense(fc_2, units=self.num_classes, activation=None)

        return fc_3

    def get_variable_scope(self):
        return 'sparse_conv_v8'
