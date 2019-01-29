import tensorflow as tf
from models.sparse_conv_clustering_base import SparseConvClusteringBase
from ops.sparse_conv_2 import *
from models.switch_model import SwitchModel
import readers.indices_calculated as ic


class BinningClusteringEpsilon(SparseConvClusteringBase):

    def __init__(self, n_space, n_space_local, n_others, n_target_dim, batch_size, max_entries, learning_rate=0.0001):
        super(BinningClusteringEpsilon, self).__init__(n_space, n_space_local, n_others, n_target_dim, batch_size,
                                                     max_entries,
                                                     learning_rate)
        self.weight_weights = []
        self.calo_type = 3
        self.max_entries=max_entries
        self.E_loss=False
        self.sum_loss=True
        self.log_loss=False


    def set_calorimeter_type(self, calo_type):
        """
        Sets the type of calo:
            1. Old calorimeter with 2679 sensors (deprecated)
            2. Homogeneous calorimeter
            3. New calorimter with 2102 sensors

        :param calo_type:
        :return:
        """
        self.calo_type = calo_type

    def construct_conversion_ops(self):
        batch_indices = np.arange(self.batch_size)
        batch_indices = np.tile(batch_indices[..., np.newaxis], reps=(1, self.max_entries))[..., np.newaxis]

        if self.calo_type==1:
            indexing_array = \
                np.concatenate((ic.x_bins[:, np.newaxis], ic.y_bins[:, np.newaxis], ic.l_bins[:, np.newaxis],
                                ic.d_indices[:, np.newaxis]),
                               axis=1)[np.newaxis, ...]
        elif self.calo_type == 2:
            indexing_array = \
                np.concatenate((ic.x_bins_homog[:, np.newaxis], ic.y_bins_homog[:, np.newaxis], ic.l_bins_homog[:, np.newaxis],
                                ic.d_indices_homog[:, np.newaxis]),
                               axis=1)[np.newaxis, ...]
        elif self.calo_type == 3:
            indexing_array = \
                np.concatenate((ic.x_bins_beta_calo[:, np.newaxis], ic.y_bins_beta_calo[:, np.newaxis], ic.l_bins_beta_calo[:, np.newaxis],
                                ic.d_indices_beta_calo[:, np.newaxis]),
                               axis=1)[np.newaxis, ...]

        indexing_array = np.tile(indexing_array, reps=[self.batch_size, 1, 1])
        self.indexing_array = np.concatenate((batch_indices, indexing_array), axis=2).astype(np.int64)

        net = construct_sparse_io_dict(self._placeholder_other_features, self._placeholder_space_features, self._placeholder_space_features_local,
                                          tf.squeeze(self._placeholder_num_entries))

        net = sparse_conv_normalise(net, log_energy=True)

        values = tf.concat((net['all_features'], net['spatial_features_local']), axis=-1)
        result = tf.scatter_nd(self.indexing_array, values, shape=(self.batch_size, 8, 8, 25, 16, 4))
        result = tf.reshape(result, [self.batch_size, 8, 8, 25, 16 * 4])

        return result

    def select_axis_indices(self, tensor, indices):
        tensors = []
        for i in indices:
            tensors.append(tensor[:, :, i][..., tf.newaxis])
        return tf.concat(tensors, axis=-1)

    def normalise_response(self,total_response):
        mean, variance = tf.nn.moments(total_response, axes=0)
        return tf.clip_by_value(mean, 0.01, 100), tf.clip_by_value(variance, 0, 100)/tf.clip_by_value(mean,0.001,100)


    def _get_loss(self):
        assert self._graph_output.shape[2] >= 2

        num_entries = tf.squeeze(self._placeholder_num_entries, axis=1)
        print('num_entries', num_entries.shape)
        energy = self._placeholder_other_features[:, :, 0]

        ###
        sqrt_energy = None
        if self.log_loss:
            sqrt_energy = tf.log(energy + 1)
        else:
            sqrt_energy = tf.sqrt(energy)

        prediction = self._graph_output
        targets = self._placeholder_targets

        maxlen = self.max_entries
        # if self.use_seeds:
        #    energy=energy[:,0:-1]
        #    targets = targets[:,0:-1,:]

        total_energy = tf.reduce_sum(energy, axis=-1)

        print('total_energy', total_energy.shape)

        energies = targets * energy[:, :, tf.newaxis]
        energy_sums = tf.reduce_sum(energies, axis=1)
        energy_a = energy_sums[:, 0]
        energy_b = energy_sums[:, 1]

        print('energy_a', energy_a.shape)

        sqrt_energies = targets * sqrt_energy[:, :, tf.newaxis]

        print('sqrt_energies', sqrt_energies.shape)

        sqrt_energy_sum = tf.reduce_sum(sqrt_energies, axis=1)
        sqrt_energy_a = sqrt_energy_sum[:, 0]
        sqrt_energy_b = sqrt_energy_sum[:, 1]

        print('sqrt_energy_sum', sqrt_energy_sum.shape)

        diff_sq = (prediction[:, :, 0:2] - targets) ** 2.
        diff_sq_a = diff_sq[:, :, 0]
        diff_sq_b = diff_sq[:, :, 1]

        print('diff_sq_a', diff_sq_a.shape)

        e_for_loss = sqrt_energies
        esum_for_loss_a = sqrt_energy_a
        esum_for_loss_b = sqrt_energy_b

        if self.E_loss:
            e_for_loss = energies
            esum_for_loss_a = energy_a
            esum_for_loss_b = energy_b

        loss_a = tf.reduce_sum(diff_sq_a * e_for_loss[:, :, 0], axis=1) / (esum_for_loss_a)
        loss_b = tf.reduce_sum(diff_sq_b * e_for_loss[:, :, 1], axis=1) / (esum_for_loss_b)

        old_loss = (tf.reduce_sum(diff_sq_a * e_for_loss[:, :, 0], axis=1) + tf.reduce_sum(
            diff_sq_b * e_for_loss[:, :, 1], axis=1)) / (esum_for_loss_a + esum_for_loss_b)

        print('loss_a', loss_a.shape)

        total_loss = (loss_a + loss_b) / 2.

        response_a = tf.reduce_sum(prediction[:, :, 0] * energy, axis=1) / energy_a
        response_b = tf.reduce_sum(prediction[:, :, 1] * energy, axis=1) / energy_b

        print('response_a', response_a.shape)

        total_response = tf.concat([response_a, response_b], axis=0)

        self.mean_resolution, self.variance_resolution = self.normalise_response(total_response)

        self.total_loss = tf.reduce_mean(total_loss)

        sqrt_response_a = tf.reduce_sum(prediction[:, :, 0] * sqrt_energies[:, :, 0], axis=1) / sqrt_energy_a
        sqrt_response_b = tf.reduce_sum(prediction[:, :, 1] * sqrt_energies[:, :, 1], axis=1) / sqrt_energy_b

        sqrt_total_response = tf.concat([sqrt_response_a, sqrt_response_b], axis=0)

        self.mean_sqrt_resolution, self.variance_sqrt_resolution = self.normalise_response(sqrt_total_response)

        if self.sum_loss:
            return self.total_loss
        return tf.reduce_mean(
            old_loss)  # + tf.reduce_mean(0.1*tf.abs(1-self.mean_resolution)+0.1*self.variance_resolution)

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

        # x = tf.layers.batch_normalization(x, training=self.is_train)
        x = tf.layers.conv3d(x, 50, [1, 1, 1], activation=tf.nn.leaky_relu, padding='same')
        x = tf.layers.conv3d(x, 50, [1, 1, 1], activation=tf.nn.leaky_relu, padding='same')
        x = tf.layers.conv3d(x, 50, [1, 1, 1], activation=tf.nn.leaky_relu, padding='same')
        x = tf.layers.conv3d(x, 32, [3, 3, 1], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 32, [1, 1, 5], activation=tf.nn.relu, padding='same')  # 8x8x25x32
        # x = tf.layers.batch_normalization(x, training=self.is_train)

        y = tf.layers.max_pooling3d(x, (2, 2, 2), 2)  # 4x4x12x32
        y = tf.layers.conv3d(y, 32, [1, 1, 4], activation=tf.nn.relu, padding='same')  # 4x4x12x32
        y = tf.layers.max_pooling3d(y, (1, 1, 3), (1,1,3))  # 4x4x4x32
        y = tf.layers.conv3d(y, 16, [2, 2, 1], activation=tf.nn.relu, padding='same')  # 4x4x4x32
        y = tf.layers.max_pooling3d(y, (2, 2, 1), (2,2,1))  # 2x2x4x16≈
        y = tf.layers.conv3d_transpose(y, 16, [2, 2, 1], [2,2,1], activation=tf.nn.relu, padding='same')  # 4x4x4x16
        y = tf.layers.conv3d_transpose(y, 16, [1, 1, 2], [1,1,2], activation=tf.nn.relu, padding='same')  # 4x4x8x16
        y = tf.layers.conv3d_transpose(y, 16, [2, 2, 1], [2,2,1], activation=tf.nn.relu, padding='same')  # 8x8x8x16
        y = tf.layers.conv3d_transpose(y, 8, [1, 1, 3], [1,1,3], activation=tf.nn.relu, padding='same')  # 4x4x8x16
        # y = tf.layers.batch_normalization(y, training=self.is_train)
        y = tf.pad(y, tf.constant([[0, 0], [0, 0], [0, 0], [0, 1], [0, 0]]))

        x = tf.layers.conv3d(x, 32, [3, 3, 1], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 24, [1, 1, 5], activation=tf.nn.relu, padding='same')

        x = tf.concat((x,y), axis=-1)

        x = tf.layers.conv3d(x, 32, [3, 3, 1], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 32, [1, 1, 5], strides=(1, 1, 1), activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 32, [3, 3, 1], activation=tf.nn.relu, padding='same')
        # x = tf.layers.batch_normalization(x, training=self.is_train)


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
