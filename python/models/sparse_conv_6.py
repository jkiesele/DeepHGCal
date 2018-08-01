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

        net = self._placeholder_all_features[:, :, 3]
        print(net.shape)
        net = tf.expand_dims(net, axis=2)
        net = net / 1000
        print(net.shape)
        reclayers = self._placeholder_all_features[:, :, 4]
        reclayers = tf.expand_dims(reclayers, axis=2) / 150
        allocalfeat = tf.concat((self._placeholder_space_features_local / 150, reclayers),
                                axis=-1)
        # net = tf.log(net+0.01)
        scaledspace = self._placeholder_space_features / 150
        scaledspace = scaledspace[:, :, 0:2]
        scaledspace = tf.concat((scaledspace, reclayers), axis=-1)

        _input = construct_sparse_io_dict(net, scaledspace,
                                          allocalfeat,
                                          tf.squeeze(self._placeholder_num_entries))

        net = _input

        net = sparse_max_pool(net, 2000)
        net = sparse_conv_2(net, num_neighbors=27, num_filters=8, n_prespace_conditions=8,
                            transform_global_space=1, transform_local_space=1)
        net = sparse_max_pool(net, 250)
        net = sparse_conv_2(net, num_neighbors=64, num_filters=32, n_prespace_conditions=8,
                            transform_global_space=4, transform_local_space=4)
        net = sparse_max_pool(net, 125)
        net = sparse_conv_2(net, num_neighbors=32, num_filters=16, n_prespace_conditions=8,
                            transform_global_space=8, transform_local_space=8)
        net = sparse_max_pool(net, 32)
        net = sparse_conv_2(net, num_neighbors=32, num_filters=64, n_prespace_conditions=8)
        net = sparse_max_pool(net, 1)

        flattened_features = sparse_merge_flat(net, combine_three=True)
        self._graph_temp = flattened_features[0]

        fc_1 = tf.layers.dense(flattened_features, units=64, activation=tf.nn.relu)
        fc_2 = tf.layers.dense(fc_1, units=32, activation=tf.nn.selu)
        fc_3 = tf.layers.dense(fc_2, units=self.num_classes, activation=None)

        return fc_3

    def get_variable_scope(self):
        return 'sparse_conv_v4'
