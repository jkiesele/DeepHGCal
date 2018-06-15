import tensorflow as tf
from models.model import Model
from hgcal_conv_conv_3d import HgCal3d
import tensorflow.contrib.slim as slim


class HgCalRecurrent3d(HgCal3d):
    def __init__(self, dim_x, dim_y, dim_z, num_input_features, batch_size, num_classes, learning_rate=0.0001):
        super(HgCalRecurrent3d, self).__init__(dim_x, dim_y, dim_z, num_input_features, batch_size, num_classes, learning_rate)

    def _find_logits(self):
        # [Batch, Height, Width, Depth, Channels]
        x = self._placeholder_inputs
        initializer = tf.random_uniform_initializer(-1, 1)
        with tf.variable_scope(self.get_variable_scope()):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(100, initializer=initializer)
            lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(80, initializer=initializer)
            with slim.arg_scope(self.arg_scope()):
                # [Batch, Depth, Height, Width, Channels]
                # Shape: [B,13,13,L,C]
                x = tf.transpose(self._placeholder_inputs, perm=[3, 0, 1, 2, 4])
                # Shape : [L,B,13,13,C]
                x = tf.reshape(x, [(self.dim_z * self.batch_size), self.dim_x, self.dim_y, self.num_input_features])
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
                x = tf.reshape(x, [self.__L, self.__B, self.dim_x, self.dim_y, 30])
                # Shape: [L, B, 13, 13, 30]
                x = tf.reshape(x, [self.__L, (self.__B * self.dim_x * self.dim_y), 30])
                # Shape : [L, (B*13*13), 30] - Time Major
                x, state = tf.nn.dynamic_rnn(lstm_cell, x,
                                             sequence_length=tf.ones(self.batch_size * self.dim_x * self.dim_y, dtype=tf.int32) * self.dim_z,
                                             dtype=tf.float32, scope="lstm_1", time_major=True)
                # Output is of shape: [L, (B*13*13, 100)
                x, state = tf.nn.dynamic_rnn(lstm_cell_2, x,
                                             sequence_length=tf.ones(self.batch_size * self.dim_x * self.dim_y, dtype=tf.int32) * self.dim_z,
                                             dtype=tf.float32, scope="lstm_2", time_major=True)

                x = x[self.dim_z - 1]

                x = tf.reshape(x, [self.batch_size, self.dim_x, self.dim_y, 80])
                x = slim.conv2d(x, 256, [1, 1], scope='e_mm', activation_fn=None)
                # Shape: [L, B, 13, 13, 1]
                x = tf.reshape(x, [self.batch_size, self.dim_x, self.dim_y, 256])
                x = tf.transpose(x, perm=[1, 2, 3, 0])

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

    def get_variable_scope(self):
        return 'h3d_recurrent_1'

    def get_human_name(self):
        return 'Recurrent 3d'