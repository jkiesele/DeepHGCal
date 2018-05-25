import tensorflow as tf
from models.model import Model
from ops.sparse_conv import *


class SparseConv(Model):
    def __init__(self, n_space, n_space_local, n_all, n_max_neighbors, batch_size, max_entries, num_classes, learning_rate=0.0001):
        self.initialized = False
        self.n_space = n_space
        self.n_space_local = n_space_local
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
        return self._placeholder_space_features,self._placeholder_space_features_local, self._placeholder_all_features, \
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
                               kernel_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width),
                               bias_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width))
        fc_2 = tf.layers.dense(fc_1, units=30, activation=tf.nn.relu,
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
        self._placeholder_space_features_local = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries, self.n_space_local])
        self._placeholder_all_features = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries, self.n_all])
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
