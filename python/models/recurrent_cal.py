import tensorflow as tf
from model import Model
slim = tf.contrib.slim


class RecurrentCal(Model):

    __error_initialize = "Need to initialize first"

    def __init__(self, B, L=55, C=24):
        """
        B = Batch
        L = Number of layers (e.g. 55)
        C  = Number of input channels (e.g. 24)

        :param placeholder_input: The input of shape [B,13,13,L,C]
        """
        self.__B = B
        self.__L = L
        self.__C = C
        self.__num_hidden_lstm = 100
        self.__learning_rate = 0.0001
        self.initialized = False

    def initialize(self):
        if self.initialized:
            print("Already initialized")
            return

        self.__construct_graphs()

    def __check_init(self):
        if not self.initialized:
            raise (RecurrentCal.__error_initialize)

    def get_summary(self):
        self.__check_init()
        return self.__graph_summary_loss

    def get_placeholders(self):
        """
        Returns a tuple of placeholders:
        1. input
        2. Regression target
        """
        self.__check_init()
        return self.__placeholder_input, self.__placeholder_regression_target

    def get_compute_graphs(self):
        """
        Return the graph op of the output of the model
        """
        self.__check_init()
        return self.__graph_output

    def get_losses(self):
        """
        Returns the loss graph op
        """
        self.__check_init()
        return self.__graph_loss

    def get_optimizer(self):
        """
        Returns the optimizer graph op
        """
        self.__check_init()
        return [self.__graph_optimizer]

    def vgg_arg_scope(self, weight_decay=0.0005):
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

    def __construct_graphs(self):
        self.initialized = True

        self.__placeholder_input = tf.placeholder(dtype=tf.float32, shape=[self.__B, 13, 13, self.__L, self.__C])
        self.__placeholder_regression_target = tf.placeholder(dtype=tf.float32, shape=[self.__B, 13, 13, self.__L])


        initializer = tf.random_uniform_initializer(-1, 1)
        lstm_cell = tf.nn.rnn_cell.LSTMCell(self.__num_hidden_lstm, initializer=initializer)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(self.__num_hidden_lstm, initializer=initializer)

        with slim.arg_scope(self.vgg_arg_scope()):
            # Shape: [B,13,13,L,C]
            x = tf.transpose(self.__placeholder_input, perm=[3, 0, 1, 2, 4])
            # Shape : [L,B,13,13,C]
            x = tf.reshape(x, [(self.__L * self.__B), 13, 13, self.__C])
            # Shape: [(L*B), 13, 13, C]
            x = slim.conv2d(x, 30, [1, 1], scope='p1_c1')
            x = slim.conv2d(x, 30, [1, 1], scope='p1_c2')
            x = slim.conv2d(x, 30, [1, 1], scope='p1_c3')
            x = slim.conv2d(x, 30, [1, 1], scope='p1_c4')
            # Shape: [(L*B), 13, 13, 30]
            x = slim.conv2d(x, 30, [5, 5], scope='p2_c1')
            x = slim.conv2d(x, 30, [5, 5], scope='p2_c2')
            x = slim.conv2d(x, 30, [5, 5], scope='p2_c3')
            x = slim.conv2d(x, 30, [5, 5], scope='p2_c4')
            # Shape: [(L*B), 13, 13, 30]
            x = tf.reshape(x, [self.__L, self.__B, 13, 13, 30])
            # Shape: [L, B, 13, 13, 30]
            x = tf.reshape(x, [self.__L, (self.__B * 13 * 13), 30])
            # Shape : [L, (B*13*13), 30] - Time Major
            x, state = tf.nn.dynamic_rnn(lstm_cell, x, sequence_length=tf.ones(self.__B * 13 * 13, dtype=tf.int32) * self.__L,
                                         dtype=tf.float32, scope="lstm_1", time_major=True)
            # Output is of shape: [L, (B*13*13, 100)
            x, state = tf.nn.dynamic_rnn(lstm_cell_2, x, sequence_length=tf.ones(self.__B * 13 * 13, dtype=tf.int32) * self.__L,
                                         dtype=tf.float32, scope="lstm_2", time_major=True)

            x = tf.reshape(x, [(self.__L * self.__B), 13, 13, 100])
            x = slim.conv2d(x, 1, [1, 1], scope='e_mm', activation_fn=None)
            # Shape: [L, B, 13, 13, 1]
            x = tf.reshape(x, [self.__L, self.__B, 13, 13])
            x = tf.transpose(x, perm=[1,2,3,0])
            # Shape: [B, 13, 13, L]

            self.__graph_output = x
            self.__graph_loss = tf.reduce_mean((self.__graph_output - self.__placeholder_regression_target) ** 2)
            self.__graph_optimizer = tf.train.AdamOptimizer(learning_rate=self.__learning_rate).minimize(self.__graph_loss)
            self.__graph_summary_loss = tf.summary.scalar('loss_complete', self.__graph_loss)


