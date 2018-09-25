import tensorflow as tf
from models.model import Model
from ops.sparse_conv import *


class SparseConvClusteringBase(Model):
    def __init__(self, n_space, n_space_local, n_others, n_target_dim, batch_size, max_entries, learning_rate=0.0001):
        self.initialized = False
        self.n_space = n_space
        self.n_space_local = n_space_local
        self.n_other_features = n_others
        self.n_target_dim = n_target_dim
        self.batch_size = batch_size
        self.max_entries = max_entries
        self.learning_rate = learning_rate

    def initialize(self):
        if self.initialized:
            print("Already initialized")
            return
        self._construct_graphs()

    def get_summary(self):
        return self.__graph_summaries

    def get_summary_validation(self):
        return self.__graph_summaries_validation

    def get_placeholders(self):
        return self._placeholder_space_features,self._placeholder_space_features_local, self._placeholder_other_features, \
               self._placeholder_targets, self._placeholder_num_entries

    def get_compute_graphs(self):
        return self.__graph_output

    def get_losses(self):
        return self.__graph_loss

    def get_optimizer(self):
        return self.__graph_optimizer

    def get_temp(self):
        return self._graph_temp

    def _compute_output(self):
        raise("Not implemented")

    def get_variable_scope(self):
        return 'sparse_conv_v1'

    def _construct_graphs(self):
        with tf.variable_scope(self.get_variable_scope()):
            self.initialized = True
            self.weight_init_width=1e-6

            self._placeholder_space_features = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries, self.n_space])
            self._placeholder_space_features_local = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries, self.n_space_local])
            self._placeholder_other_features = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries, self.n_other_features])
            self._placeholder_targets = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries, self.n_target_dim])
            self._placeholder_num_entries = tf.placeholder(dtype=tf.int64, shape=[self.batch_size, 1])

            self.__graph_output = self._compute_output()

            # self._graph_temp = tf.nn.softmax(self.__graph_logits)

            self.__graph_loss = tf.reduce_sum((self.__graph_output - self._placeholder_targets)**2 * tf.cast(tf.sequence_mask(tf.squeeze(self._placeholder_num_entries, axis=1), maxlen=self.max_entries)[:,:,tf.newaxis], tf.float32))/ tf.cast(tf.reduce_sum(self._placeholder_num_entries), tf.float32)

            self.__graph_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.__graph_loss)

            # Repeating, maybe there is a better way?
            self.__graph_summary_loss = tf.summary.scalar('Loss', self.__graph_loss)
            self.__graph_summaries = tf.summary.merge([self.__graph_summary_loss])

            self.__graph_summary_loss_validation = tf.summary.scalar('Validation Loss', self.__graph_loss)
            self.__graph_summaries_validation = tf.summary.merge([self.__graph_summary_loss_validation])
