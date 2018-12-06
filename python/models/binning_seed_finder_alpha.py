import tensorflow as tf
from models.sparse_conv_clustering_base import SparseConvClusteringBase
from ops.sparse_conv_2 import *
from models.switch_model import SwitchModel
import readers.indices_calculated as ic


class BinningSeedFinderAlpha(SparseConvClusteringBase):

    def __init__(self, n_space, n_space_local, n_others, n_target_dim, batch_size, max_entries, learning_rate=0.0001):
        super(BinningSeedFinderAlpha, self).__init__(n_space, n_space_local, n_others, n_target_dim, batch_size,
                                                                 max_entries,
                                                                 learning_rate)
        self.weight_weights = []
        self.AdMat = None
        self.use_seeds = True

        self.fixed_seeds = None
        self.is_training = True
        self.log_energy = False
        self.log_loss = False
        self.seed_talk = True
        self.sqrt_energy = True
        self.loss_energy_function = tf.sqrt
        self.min_loss_mode = False

    def set_input_energy_log(self, log_energy):
        self.log_energy = log_energy

    def set_loss_energy_function(self, loss_energy_funtion):
        """
        Sets the loss energy function to use for loss scaling

        :param loss_energy_function: The function to call on energy values
        :return:
        """
        self.loss_energy_function = loss_energy_funtion

    def set_training(self, is_training):
        self.is_training = is_training

    def set_seed_talk(self, seed_talk):
        self.seed_talk=seed_talk

    def make_placeholders(self):
        self._placeholder_space_features = tf.placeholder(dtype=tf.float32,
                                                          shape=[self.batch_size, self.max_entries, self.n_space])
        self._placeholder_space_features_local = tf.placeholder(dtype=tf.float32,
                                                                shape=[self.batch_size, self.max_entries,
                                                                       self.n_space_local])
        self._placeholder_other_features = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries,
                                                                                   self.n_other_features])
        self._placeholder_targets = tf.placeholder(dtype=tf.float32,
                                                   shape=[self.batch_size, self.max_entries, self.n_target_dim])
        self._placeholder_num_entries = tf.placeholder(dtype=tf.int64, shape=[self.batch_size, 1])
        self._placeholder_seed_indices = tf.placeholder(dtype=tf.int64, shape=[self.batch_size, 2])

    def get_placeholders(self):
        return self._placeholder_space_features, self._placeholder_space_features_local, self._placeholder_other_features, \
               self._placeholder_targets, self._placeholder_num_entries, self._placeholder_seed_indices

    def set_min_loss_mode(self, min_loss_mode):
        self.min_loss_mode = min_loss_mode

    def make_seed_targets(self):
        num_sensors = self._placeholder_space_features.shape[1]
        indices_shower_1 = tf.concat((tf.range(self.batch_size, dtype=tf.int64)[..., tf.newaxis], self._placeholder_seed_indices[:, 0][..., tf.newaxis]), axis=1)
        indices_shower_2 = tf.concat((tf.range(self.batch_size, dtype=tf.int64)[..., tf.newaxis], self._placeholder_seed_indices[:, 1][..., tf.newaxis]), axis=1)
        updates = tf.ones(shape=self.batch_size, dtype=tf.int64)
        shape = tf.constant([self.batch_size, num_sensors], dtype=tf.int64)

        target_column_1 = tf.scatter_nd(indices_shower_1, updates, shape)[..., tf.newaxis]
        target_column_2 = tf.scatter_nd(indices_shower_2, updates, shape)[..., tf.newaxis]

        target_column_3 = 1 - tf.minimum(target_column_1+target_column_2, 1)

        return tf.concat((target_column_1, target_column_2, target_column_3), axis=2)

    def _get_loss(self):
        assert self._graph_output.shape[2] == 3

        num_entries = tf.squeeze(self._placeholder_num_entries, axis=1)
        print('num_entries', num_entries.shape)

        prediction = self._graph_output
        targets = self.make_seed_targets()
        # self._graph_temp = tf.reduce_sum(targets,axis=1)

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction[:, :, 0], labels=targets[:,:,0])) +\
               tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction[:, :, 1], labels=targets[:,:,1]))

        # self._graph_accuracy = tf.reduce_mean(tf.cast((tf.argmax(tf.nn.softmax(prediction), axis=1)[:, 0:2] == self._placeholder_seed_indices[:, 0:2]), tf.float32))

        self._graph_accuracy = tf.reduce_mean(tf.concat((
            tf.cast(tf.equal(tf.argmax(prediction[:, :, 0], axis=1), self._placeholder_seed_indices[:, 0]), tf.float32),
            tf.cast(tf.equal(tf.argmax(prediction[:, :, 1], axis=1), self._placeholder_seed_indices[:, 1]), tf.float32)), axis=0
        ))

        return loss


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
        x = tf.layers.conv3d(x, 32, [1, 1, 5], activation=tf.nn.relu, padding='same')

        x = tf.layers.conv3d(x, 32, [3, 3, 1], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 32, [1, 1, 5], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 32, [3, 3, 1], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 32, [1, 1, 5], strides=(1,1,1), activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 32, [3, 3, 1], activation=tf.nn.relu, padding='same')

        x = tf.layers.conv3d(x, 32, [1, 1, 5], strides=(1,1,1), activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 32, [3, 3, 1], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 32, [1, 1, 5], strides=(1,1,1), activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 32, [2, 2, 1], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 48, [1, 1, 3], activation=tf.nn.relu, padding='same')


        x = tf.reshape(x, [self.batch_size, 8, 8, 25, 16, 3])
        output = tf.gather_nd(x, self.indexing_array)

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

            extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_ops):
                self._graph_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self._graph_loss)

            # Repeating, maybe there is a better way?
            self._graph_summary_loss = tf.summary.scalar('Loss', self._graph_loss)
            self._graph_summary_accu = tf.summary.scalar('Accuracy', self._graph_accuracy)

            self._graph_summaries = tf.summary.merge([self._graph_summary_loss, self._graph_summary_accu])

            self._graph_summary_loss_validation = tf.summary.scalar('Validation Loss', self._graph_loss)
            self._graph_summary_accu_validation = tf.summary.scalar('Validation Accuracy', self._graph_accuracy)
            self._graph_summaries_validation = tf.summary.merge([self._graph_summary_loss_validation, self._graph_summary_accu_validation])



    def construct_conversion_ops(self):
        batch_indices = np.arange(self.batch_size)
        batch_indices = np.tile(batch_indices[..., np.newaxis], reps=(1, 2679))[..., np.newaxis]
        indexing_array = \
        np.concatenate((ic.x_bins[:, np.newaxis], ic.y_bins[:, np.newaxis], ic.l_bins[:, np.newaxis], ic.d_indices[:, np.newaxis]),
                       axis=1)[np.newaxis, ...]
        indexing_array = np.tile(indexing_array, reps=[self.batch_size, 1, 1])
        self.indexing_array = np.concatenate((batch_indices, indexing_array), axis=2).astype(np.int64)

        values = tf.concat((self._placeholder_other_features, self._placeholder_space_features_local), axis=-1)
        result = tf.scatter_nd(self.indexing_array, values, shape=(self.batch_size, 8, 8, 25, 16, 4))
        result = tf.reshape(result, [self.batch_size, 8, 8, 25, 16*4])

        return result

    def get_losses(self):
        print("Hello, world!")
        return self._graph_loss

