import tensorflow as tf
from models.sparse_conv_clustering_base import SparseConvClusteringBase
from ops.sparse_conv import *
from models.switch_model import SwitchModel
import readers.indices_calculated as ic


class BinningClusteringEpsilon(SparseConvClusteringBase):

    def __init__(self, n_space, n_space_local, n_others, n_target_dim, batch_size, max_entries, learning_rate=0.0001):
        super(BinningClusteringEpsilon, self).__init__(n_space, n_space_local, n_others, n_target_dim, batch_size,
                                                     max_entries,
                                                     learning_rate)
        self.weight_weights = []
        self.homogeneous_calorimeter = False

    def set_homogeneous_calorimeter(self, homogeneous_calorimeter):
        self.homogeneous_calorimeter = homogeneous_calorimeter

    def construct_conversion_ops(self):
        batch_indices = np.arange(self.batch_size)
        batch_indices = np.tile(batch_indices[..., np.newaxis], reps=(1, 2679))[..., np.newaxis]

        if not self.homogeneous_calorimeter:
            indexing_array = \
                np.concatenate((ic.x_bins[:, np.newaxis], ic.y_bins[:, np.newaxis], ic.l_bins[:, np.newaxis],
                                ic.d_indices[:, np.newaxis]),
                               axis=1)[np.newaxis, ...]
        else :
            indexing_array = \
                np.concatenate((ic.x_bins_homog[:, np.newaxis], ic.y_bins_homog[:, np.newaxis], ic.l_bins_homog[:, np.newaxis],
                                ic.d_indices_homog[:, np.newaxis]),
                               axis=1)[np.newaxis, ...]

        indexing_array = np.tile(indexing_array, reps=[self.batch_size, 1, 1])
        self.indexing_array = np.concatenate((batch_indices, indexing_array), axis=2).astype(np.int64)

        values = tf.concat((self._placeholder_other_features, self._placeholder_space_features_local), axis=-1)
        result = tf.scatter_nd(self.indexing_array, values, shape=(self.batch_size, 8, 8, 25, 16, 4))
        result = tf.reshape(result, [self.batch_size, 8, 8, 25, 16 * 4])

        return result

    def select_axis_indices(self, tensor, indices):
        tensors = []
        for i in indices:
            tensors.append(tensor[:, :, i][..., tf.newaxis])
        return tf.concat(tensors, axis=-1)

    def _get_loss(self):
        assert self._graph_output.shape[2] == 3

        num_entries = tf.squeeze(self._placeholder_num_entries, axis=1)
        print('num_entries', num_entries.shape)
        energy = self._placeholder_other_features[:, :, 0]
        sqrt_energy = tf.sqrt(energy)

        prediction = self._graph_output
        targets = self._placeholder_targets

        maxlen = self.max_entries
        # if self.use_seeds:
        #    energy=energy[:,0:-1]
        #    targets = targets[:,0:-1,:]

        diff_sq_1 = (prediction[:, :, 0:2] - targets) ** 2 * tf.cast(
            tf.sequence_mask(num_entries, maxlen=self.max_entries)[:, :,
            tf.newaxis], tf.float32) * sqrt_energy[:, :, tf.newaxis]
        diff_sq_1 = tf.reduce_sum(diff_sq_1, axis=[-1, -2]) / tf.reduce_sum(sqrt_energy, axis=-1)
        loss_unreduced_1 = (diff_sq_1 / tf.cast(num_entries, tf.float32)) * tf.cast(
            num_entries != 0, tf.float32)

        diff_sq_2 = (prediction[:, :, 0:2] - (1 - targets)) ** 2 * tf.cast(
            tf.sequence_mask(num_entries, maxlen=self.max_entries)[:, :,
            tf.newaxis], tf.float32) * sqrt_energy[:, :, tf.newaxis]
        diff_sq_2 = tf.reduce_sum(diff_sq_2, axis=[-1, -2]) / tf.reduce_sum(sqrt_energy, axis=-1)
        loss_unreduced_2 = (diff_sq_2 / tf.cast(num_entries, tf.float32)) * tf.cast(
            num_entries != 0, tf.float32)

        shower_indices = tf.argmin(
            tf.concat((loss_unreduced_1[:, tf.newaxis], loss_unreduced_2[:, tf.newaxis]), axis=-1), axis=-1)

        condition_1 = tf.to_float(tf.equal((tf.to_float(shower_indices)[:, tf.newaxis, tf.newaxis]), 0.0))
        condition_2 = tf.to_float(tf.equal((tf.to_float(shower_indices)[:, tf.newaxis, tf.newaxis]), 1.0))
        sorted_target = targets * condition_1 + (1 - targets) * condition_2

        # + (1-targets) * tf.cast(shower_indices[:,tf.newaxis,tf.newaxis]==1, tf.float32)

        perf1 = tf.reduce_sum(prediction[:, :, 0] * energy, axis=[-1]) / tf.reduce_sum(sorted_target[:, :, 0] * energy,
                                                                                       axis=[-1])
        perf2 = tf.reduce_sum(prediction[:, :, 1] * energy, axis=[-1]) / tf.reduce_sum(sorted_target[:, :, 1] * energy,
                                                                                       axis=[-1])

        self._histogram_resolution = tf.summary.histogram("histogram_resolution_tboard",
                                                          tf.concat((perf1, perf2), axis=0))

        mean_resolution, variance_resolution = tf.nn.moments(tf.concat((perf1, perf2), axis=0), axes=0)

        self.mean_resolution = tf.clip_by_value(mean_resolution, 0.2, 2)
        self.variance_resolution = tf.clip_by_value(variance_resolution, 0, 1) / tf.clip_by_value(mean_resolution, 0.2,
                                                                                                  2)

        return tf.reduce_mean(loss_unreduced_1) * 1000.

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

        x = tf.layers.conv3d(0.001 * x, 50, [1, 1, 1], activation=tf.nn.leaky_relu, padding='same')
        x = tf.layers.conv3d(x, 50, [1, 1, 1], activation=tf.nn.leaky_relu, padding='same')
        x = tf.layers.conv3d(x, 50, [1, 1, 1], activation=tf.nn.leaky_relu, padding='same')
        x = tf.layers.conv3d(x, 32, [3, 3, 1], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 32, [1, 1, 5], activation=tf.nn.relu, padding='same')  # 8x8x25x32
        x = tf.layers.batch_normalization(x, training=self.is_train)

        y = tf.layers.max_pooling3d(x, (2, 2, 2), 2)  # 4x4x12x32
        y = tf.layers.conv3d(y, 32, [1, 1, 4], activation=tf.nn.relu, padding='same')  # 4x4x12x32
        y = tf.layers.max_pooling3d(y, (1, 1, 3), (1,1,3))  # 4x4x4x32
        y = tf.layers.conv3d(y, 16, [2, 2, 1], activation=tf.nn.relu, padding='same')  # 4x4x4x32
        y = tf.layers.max_pooling3d(y, (2, 2, 1), (2,2,1))  # 2x2x4x16≈
        y = tf.layers.conv3d_transpose(y, 16, [2, 2, 1], [2,2,1], activation=tf.nn.relu, padding='same')  # 4x4x4x16
        y = tf.layers.conv3d_transpose(y, 16, [1, 1, 2], [1,1,2], activation=tf.nn.relu, padding='same')  # 4x4x8x16
        y = tf.layers.conv3d_transpose(y, 16, [2, 2, 1], [2,2,1], activation=tf.nn.relu, padding='same')  # 8x8x8x16
        y = tf.layers.conv3d_transpose(y, 8, [1, 1, 3], [1,1,3], activation=tf.nn.relu, padding='same')  # 4x4x8x16
        x = tf.layers.batch_normalization(x, training=self.is_train)
        y = tf.pad(y, tf.constant([[0, 0], [0, 0], [0, 0], [0, 1], [0, 0]]))

        x = tf.layers.conv3d(x, 32, [3, 3, 1], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 24, [1, 1, 5], activation=tf.nn.relu, padding='same')

        x = tf.concat((x,y), axis=-1)

        x = tf.layers.conv3d(x, 32, [3, 3, 1], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 32, [1, 1, 5], strides=(1, 1, 1), activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 32, [3, 3, 1], activation=tf.nn.relu, padding='same')
        x = tf.layers.batch_normalization(x, training=self.is_train)


        x = tf.layers.conv3d(x, 32, [1, 1, 5], strides=(1, 1, 1), activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 32, [3, 3, 1], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 32, [1, 1, 5], strides=(1, 1, 1), activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 32, [2, 2, 1], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 48, [1, 1, 3], activation=tf.nn.relu, padding='same')

        x = tf.reshape(x, [self.batch_size, 8, 8, 25, 16, 3])
        output = tf.gather_nd(x, self.indexing_array)
        output = tf.nn.softmax(output)

        self._graph_temp = output

        return output

    def get_variable_scope(self):
        return 'sparse_conv_clustering_spatial1'

    def _construct_graphs(self):
        with tf.variable_scope(self.get_variable_scope()):
            self.initialized = True
            self.weight_init_width = 1e-6

            self.make_placeholders()

            self._graph_output = self._compute_output()

            # self._graph_temp = tf.nn.softmax(self.__graph_logits)

            self._graph_loss = self._get_loss()

            self._graph_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self._graph_loss)

            # Repeating, maybe there is a better way?
            self._graph_summary_loss = tf.summary.scalar('loss', self._graph_loss)
            self._graph_summaries = tf.summary.merge(
                [self._graph_summary_loss, tf.summary.scalar('mean-res', self.mean_resolution),
                 tf.summary.scalar('variance-res', self.variance_resolution)])

            self._graph_summary_loss_validation = tf.summary.scalar('Validation Loss', self._graph_loss)
            self._graph_summaries_validation = tf.summary.merge([self._graph_summary_loss_validation])

    def get_losses(self):
        return self._graph_loss
