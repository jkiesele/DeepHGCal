import tensorflow as tf
from models.model import Model
from ops.sparse_conv import *
import inspect
import sys
import importlib


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
        self.use_seeds=False
        self.is_train = tf.placeholder(tf.bool, name="is_train");
        

        
    def initialize(self):
        if self.initialized:
            print("Already initialized")
            return
        self._construct_graphs()

    def get_summary(self):
        return self._graph_summaries

    def get_performance_measures(self):
        raise RuntimeError("Unimplemented")

    def get_summary_validation(self):
        return self._graph_summaries_validation

    def get_placeholders(self):
        return self._placeholder_space_features,self._placeholder_space_features_local, self._placeholder_other_features, \
               self._placeholder_targets, self._placeholder_num_entries

    def get_placeholders_dict(self):
        placeholders = dict()
        placeholders['features_space'] = self._placeholder_space_features
        placeholders['features_space_local'] = self._placeholder_space_features_local
        placeholders['features_others'] = self._placeholder_other_features
        placeholders['targets'] = self._placeholder_targets
        placeholders['num_entries'] = self._placeholder_num_entries
        return placeholders

    def get_compute_graphs(self):
        return self._graph_output

    def get_losses(self):
        return self._graph_loss

    def get_optimizer(self):
        return self._graph_optimizer

    def get_temp(self):
        return self._graph_temp

    def _compute_output(self):
        raise("Not implemented")

    def get_variable_scope(self):
        return 'sparse_conv_v1'


    def _get_loss(self):
        diff_sq = (self._graph_output - self._placeholder_targets) ** 2 * tf.cast(
            tf.sequence_mask(tf.squeeze(self._placeholder_num_entries, axis=1), maxlen=self.max_entries)[:, :,
            tf.newaxis], tf.float32)
        diff_sq = tf.reduce_sum(diff_sq, axis=[-1, -2])

        loss_unreduced = (diff_sq / tf.cast(self._placeholder_num_entries, tf.float32)) * tf.cast(
            self._placeholder_num_entries != 0, tf.float32)

        return tf.reduce_mean(loss_unreduced)

    def make_placeholders(self):
        self._placeholder_space_features = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries, self.n_space])
        self._placeholder_space_features_local = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries, self.n_space_local])
        self._placeholder_other_features = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries, self.n_other_features])
        self._placeholder_targets = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries, self.n_target_dim])
        self._placeholder_num_entries = tf.placeholder(dtype=tf.int64, shape=[self.batch_size, 1])


    def _construct_graphs(self):
        with tf.variable_scope(self.get_variable_scope()):
            self.initialized = True
            self.weight_init_width=1e-6

            self.make_placeholders()

            self._graph_output = self._compute_output()

            # self._graph_temp = tf.nn.softmax(self.__graph_logits)

            self._graph_loss = self._get_loss()

            self._graph_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self._graph_loss)

            # Repeating, maybe there is a better way?
            self._graph_summary_loss = tf.summary.scalar('Loss', self._graph_loss)
            self._graph_summaries = tf.summary.merge([self._graph_summary_loss])

            self._graph_summary_loss_validation = tf.summary.scalar('Validation Loss', self._graph_loss)
            self._graph_summaries_validation = tf.summary.merge([self._graph_summary_loss_validation])

    def get_code(self):
        return inspect.getsource(importlib.import_module(self.__module__))

