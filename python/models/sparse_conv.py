import tensorflow as tf
from models.model import Model
from ops.sparse_conv import sparse_conv_2
from ops.sparse_conv import sparse_conv_bare
import numpy as np


def printLayerStuff(l, desc_str):
    return l
    l=tf.Print(l,[l], desc_str+" ",summarize=2000)
    return l

class SparseConv(Model):
    def __init__(self, n_space, n_all, n_max_neighbors, batch_size, max_entries, num_classes, learning_rate=0.0001):
        self.initialized = False
        self.n_space = n_space
        self.n_all = n_all
        self.n_max_neighbors = n_max_neighbors
        self.batch_size = batch_size
        self.max_entries = max_entries
        self.num_classes = num_classes
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
        return self._placeholder_space_features, self._placeholder_all_features, self._placeholder_neighbors_matrix, \
               self._placeholder_labels, self._placeholder_num_entries

    def get_compute_graphs(self):
        return self.__graph_logits, self.__graph_prediction

    def get_losses(self):
        return self.__graph_loss

    def get_optimizer(self):
        return self.__graph_optimizer

    def get_accuracy(self):
        return self.__accuracy

    def get_confusion_matrix(self):
        return self.__confusion_matrix

    def get_temp(self):
        return self.__graph_temp

    def _find_logits(self):
        layer_1_out, layer_1_out_spatial = sparse_conv_bare(self._placeholder_space_features,
                                                            self._placeholder_all_features,
                                                            self._placeholder_neighbors_matrix, 15)
        layer_1_out = printLayerStuff(layer_1_out, "layer_1_out")
        layer_1_out_spatial = printLayerStuff(layer_1_out_spatial, "layer_1_out_spatial")

        layer_2_out, layer_2_out_spatial = sparse_conv_bare(layer_1_out_spatial, layer_1_out,
                                                            self._placeholder_neighbors_matrix, 20)
        layer_3_out, layer_3_out_spatial = sparse_conv_bare(layer_2_out_spatial, layer_2_out,
                                                            self._placeholder_neighbors_matrix, 25)
        layer_4_out, layer_4_out_spatial = sparse_conv_2(layer_3_out_spatial, layer_3_out,
                                                         self._placeholder_neighbors_matrix, 30)
        layer_4_out = printLayerStuff(layer_4_out, "layer_4_out")
        layer_4_out_spatial = printLayerStuff(layer_4_out_spatial, "layer_4_out_spatial")

        layer_5_out, layer_5_out_spatial = sparse_conv_2(layer_4_out_spatial, layer_4_out,
                                                         self._placeholder_neighbors_matrix, 35)
        layer_6_out, layer_6_out_spatial = sparse_conv_2(layer_5_out_spatial, layer_5_out,
                                                         self._placeholder_neighbors_matrix, 40)

        # TODO: Verify this code
        squeezed_num_entries = tf.squeeze(self._placeholder_num_entries, axis=1)
        mask = tf.cast(tf.expand_dims(tf.sequence_mask(squeezed_num_entries, maxlen=self.max_entries), axis=2),
                       tf.float32)

        nonzeros = tf.count_nonzero(mask, axis=1, dtype=tf.float32)

        # jsut safety measure - should not be necessary once done
        flattened_features = tf.clip_by_value(tf.reduce_sum(layer_6_out * mask, axis=1) / nonzeros,
                                              clip_value_min=-1e9,
                                              clip_value_max=1e9)

        fc_1 = tf.layers.dense(flattened_features, units=100, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width),
                               bias_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width))
        fc_2 = tf.layers.dense(fc_1, units=100, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width),
                               bias_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width))
        fc_3 = tf.layers.dense(fc_2, units=self.num_classes, activation=None,
                               kernel_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width),
                               bias_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width))

        return fc_3

    def _construct_graphs(self):
        self.initialized = True
        self.weight_init_width=1e-6

        self._placeholder_space_features = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries, self.n_space])
        self._placeholder_all_features = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries, self.n_all])
        self._placeholder_neighbors_matrix = tf.placeholder(dtype=tf.int64, shape=[self.batch_size, self.max_entries, self.n_max_neighbors])
        self._placeholder_labels = tf.placeholder(dtype=tf.int64, shape=[self.batch_size, self.num_classes])
        self._placeholder_num_entries = tf.placeholder(dtype=tf.int64, shape=[self.batch_size, 1])

        self.__graph_logits = self._find_logits()
        self.__graph_temp = tf.nn.softmax(self.__graph_logits)

        self.__graph_prediction = tf.argmax(self.__graph_logits, axis=1)

        argmax_labels = tf.argmax(self._placeholder_labels, axis=1)

        self.__accuracy = tf.reduce_mean(tf.cast(tf.equal(argmax_labels, self.__graph_prediction), tf.float32)) * 100
        self.__confusion_matrix = tf.confusion_matrix(labels=argmax_labels, predictions=self.__graph_prediction, num_classes=self.num_classes)
        self.__graph_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.__graph_logits, labels=self._placeholder_labels))

        self.__graph_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.__graph_loss)

        # Repeating, maybe there is a better way?
        self.__graph_summary_loss = tf.summary.scalar('Loss', self.__graph_loss)
        self.__graph_summary_accuracy = tf.summary.scalar('Accuracy', self.__accuracy)
        self.__graph_summaries = tf.summary.merge([self.__graph_summary_loss, self.__graph_summary_accuracy])

        self.__graph_summary_loss_validation = tf.summary.scalar('Validation Loss', self.__graph_loss)
        self.__graph_summary_accuracy_validation = tf.summary.scalar('Validation Accuracy', self.__accuracy)
        self.__graph_summaries_validation = tf.summary.merge([self.__graph_summary_loss_validation, self.__graph_summary_accuracy_validation])
