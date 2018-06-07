import tensorflow as tf
from models.model import Model
import tensorflow.contrib.slim as slim


class HgCal3d(Model):
    def __init__(self, dim_x, dim_y, dim_z, num_input_features, batch_size, num_classes, learning_rate=0.0001):
        self.initialized = False
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.num_input_features = num_input_features

    def get_variable_scope(self):
        return 'h3d_conv_1'

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
        return self._placeholder_inputs, self._placeholder_labels

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

    def arg_scope(self, weight_decay=0.0005):
        """Defines the RecurrentCal arg scope.
        Args:
          weight_decay: The l2 regularization coefficient.
        Returns:
          An arg_scope.
        """
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                return arg_sc

    def _find_logits(self):
        x = self._placeholder_inputs
        with tf.variable_scope(self.get_variable_scope()):
            with slim.arg_scope(self.arg_scope()):
                x = slim.conv3d(x, 32, [1, 1, 1], activation_fn=tf.nn.relu, scope='p1_c1', weights_initializer=tf.keras.initializers.lecun_uniform())
                x = slim.conv3d(x, 12, [1, 1, 1], activation_fn=tf.nn.relu, scope='p1_c2', weights_initializer=tf.keras.initializers.lecun_uniform())
                x = slim.batch_norm(x)
                x = slim.conv3d(x, 32, [5, 5, 5], activation_fn=tf.nn.leaky_relu, scope='p1_c3')
                x = slim.conv3d(x, 8, [5, 5, 5], activation_fn=tf.nn.leaky_relu, scope='p1_c4')
                x = slim.conv3d(x, 8, [5, 5, 5], activation_fn=tf.nn.leaky_relu, scope='p1_c5')
                x = slim.conv3d(x, 8, [5, 5, 5], activation_fn=tf.nn.leaky_relu, scope='p1_c6')

                x = slim.max_pool3d(x, [6, 2, 2], scope='p2_m1')
                x = slim.conv3d(x, 8, [3, 3, 3], activation_fn=tf.nn.leaky_relu, scope='p2_c1')
                x = slim.conv3d(x, 8, [3, 3, 3], activation_fn=tf.nn.leaky_relu, scope='p2_c2')
                x = slim.conv3d(x, 8, [3, 3, 3], activation_fn=tf.nn.leaky_relu, scope='p2_c3')

                # x should be [B, H, W, D, 8]

                flattened_features = tf.reshape(x, (self.batch_size, -1))

                fc_1 = tf.layers.dense(flattened_features, units=1024, activation=tf.nn.relu,
                                       kernel_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width),
                                       bias_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width))
                fc_2 = tf.layers.dense(fc_1, units=1024, activation=tf.nn.relu,
                                       kernel_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width),
                                       bias_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width))
                fc_3 = tf.layers.dense(fc_2, units=self.num_classes, activation=None,
                                       kernel_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width),
                                       bias_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width))

                return fc_3

    def _construct_graphs(self):
        self.initialized = True
        self.weight_init_width=1e-6

        self._placeholder_inputs = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.dim_x, self.dim_y, self.dim_z, self.num_input_features])
        self._placeholder_labels = tf.placeholder(dtype=tf.int64, shape=[self.batch_size, self.num_classes])

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
