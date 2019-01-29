from models.binning_cluster_epsilon import BinningClusteringEpsilon
import tensorflow as tf
import numpy as np
import readers.indices_calculated as ic
from ops.sparse_conv_2 import *


class BinningClusteringZeta(BinningClusteringEpsilon):

    def __init__(self, n_space, n_space_local, n_others, n_target_dim, batch_size, max_entries, learning_rate=0.0001):
        super(BinningClusteringZeta, self).__init__(n_space, n_space_local, n_others, n_target_dim, batch_size,
                                                     max_entries,
                                                     learning_rate)

    def _compute_output(self):
        # # nl_all = tf.layers.dense(tf.scalar_mul(0.001, self._placeholder_all_features), units=8, activation=tf.nn.relu)
        # # nl_all = tf.layers.dense(nl_all, units=8, activation=tf.nn.relu)
        # # nl_all = tf.layers.dense(nl_all, units=8, activation=tf.nn.relu)
        #
        # # TODO: Remove it later after regenerating the data, this only picks energy (or do something similar)
        # net = self._placeholder_all_features
        # net = tf.concat((net, self._placeholder_space_features_local), axis=2)

        # TODO: Will cause problems with batch size of 1

        binned_input = self.construct_conversion_ops()

        # 8x8x25x64
        x = binned_input
        x = tf.layers.batch_normalization(x, training=self.is_train, center=False)
        x = tf.layers.conv3d(x, 36, [1, 1, 1], activation=tf.nn.leaky_relu, padding='same')
        x = tf.layers.conv3d(x, 36, [1, 1, 1], activation=tf.nn.leaky_relu, padding='same')
        x = tf.layers.conv3d(x, 32, [2, 2, 1], strides=[2, 2, 1], activation=tf.nn.relu, padding='same')  # 8x8x20x32
        x = tf.layers.conv3d(x, 32, [1, 1, 2], strides=[1, 1, 2], activation=tf.nn.relu, padding='same')  # 8x8x10x32
        x = tf.layers.batch_normalization(x, training=self.is_train)
        x = tf.layers.conv3d(x, 50, [2, 2, 1], strides=[2, 2, 1], activation=tf.nn.relu, padding='same')  # 4x4x10x50
        x = tf.layers.conv3d(x, 50, [1, 1, 2], strides=[1, 1, 2], activation=tf.nn.relu, padding='same')  # 4x4x5x50
        x = tf.layers.conv3d(x, 50, [2, 2, 1], strides=[1, 1, 1], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 50, [1, 1, 2], strides=[1, 1, 1], activation=tf.nn.relu, padding='same')
        x = tf.layers.batch_normalization(x, training=self.is_train)
        x = tf.layers.conv3d(x, 50, [2, 2, 1], strides=[1, 1, 1], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 50, [1, 1, 2], strides=[1, 1, 1], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 52, [2, 2, 1], strides=[1, 1, 1], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 52, [1, 1, 2], strides=[1, 1, 1], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 52, [2, 2, 1], strides=[1, 1, 1], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d_transpose(x, 64, [1, 1, 2], strides=[1, 1, 2], activation=tf.nn.relu, padding='same')  # 4x4x10x32
        x = tf.layers.conv3d_transpose(x, 64, [2, 2, 1], strides=[2, 2, 1], activation=tf.nn.relu, padding='same')  # 8x8x10x32
        x = tf.layers.conv3d_transpose(x, 64, [1, 1, 2], strides=[1, 1, 2], activation=tf.nn.relu, padding='same')  # 8x8x20x32
        x = tf.layers.conv3d(x, 48, [1, 1, 1], strides=[1, 1, 1], activation=tf.nn.relu, padding='same')  # 8x8x20x12

        x = tf.reshape(x, [self.batch_size, 8, 8, 20, 16, 3])
        output = tf.gather_nd(x, self.indexing_array2)
        output = tf.nn.softmax(output)

        self._graph_temp = output

        return output

    def construct_conversion_ops(self):
        batch_indices = np.arange(self.batch_size)
        batch_indices = np.tile(batch_indices[..., np.newaxis], reps=(1, self.max_entries))[..., np.newaxis]


        indexing_array = \
            np.concatenate((ic.x_bins_beta_calo_16[:, np.newaxis], ic.y_bins_beta_calo_16[:, np.newaxis], ic.l_bins_beta_calo_16[:, np.newaxis],
                            ic.d_indices_beta_calo_16[:, np.newaxis]),
                           axis=1)[np.newaxis, ...]

        indexing_array = np.tile(indexing_array, reps=[self.batch_size, 1, 1])
        self.indexing_array = np.concatenate((batch_indices, indexing_array), axis=2).astype(np.int64)


        indexing_array2 = \
            np.concatenate((ic.x_bins_beta_calo[:, np.newaxis], ic.y_bins_beta_calo[:, np.newaxis], ic.l_bins_beta_calo[:, np.newaxis],
                            ic.d_indices_beta_calo[:, np.newaxis]),
                           axis=1)[np.newaxis, ...]

        indexing_array2 = np.tile(indexing_array2, reps=[self.batch_size, 1, 1])
        self.indexing_array2 = np.concatenate((batch_indices, indexing_array2), axis=2).astype(np.int64)

        net = construct_sparse_io_dict(self._placeholder_other_features, self._placeholder_space_features, self._placeholder_space_features_local,
                                          tf.squeeze(self._placeholder_num_entries))

        net = sparse_conv_normalise(net, log_energy=True)

        values = tf.concat((net['all_features'], net['spatial_features_local']), axis=-1)
        result = tf.scatter_nd(self.indexing_array, values, shape=(self.batch_size, 16, 16, 20, 4, 4))
        result = tf.reshape(result, [self.batch_size, 16, 16, 20, 4 * 4])

        return result
