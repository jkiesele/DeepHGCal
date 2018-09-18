import tensorflow as tf
from models.model import Model
from models.hgcal_conv_conv_3d import HgCal3d
import tensorflow.contrib.slim as slim


class HgCal3dEnergy(HgCal3d):
    def __init__(self, dim_x, dim_y, dim_z, num_input_features, batch_size, num_classes, learning_rate=0.0001):
        super(HgCal3dEnergy, self).__init__(dim_x, dim_y, dim_z, num_input_features, batch_size, num_classes, learning_rate)

    def _find_logits(self):
        # [Batch, Height, Width, Depth, Channels]
        x = self._placeholder_inputs
        with tf.variable_scope(self.get_variable_scope()):
            with slim.arg_scope(self.arg_scope()):
                # [Batch, Depth, Height, Width, Channels]
                # x = tf.transpose(x, perm=[0,3,1,2,4])

                useful_channels = [0, 7, 14, 21, 28, 35]
                x = x[:, :, :, :, 0:42:7]

                x = tf.layers.conv3d(0.001 * x, 50, [1, 1, 1], activation=tf.nn.leaky_relu, padding='same')
                x = tf.layers.conv3d(x, 50, [1, 1, 1], activation=tf.nn.leaky_relu, padding='same')
                x = tf.layers.conv3d(x, 50, [1, 1, 1], activation=tf.nn.leaky_relu, padding='same')

                x = tf.layers.conv3d(x, 25, [3, 3, 1], activation=tf.nn.relu, padding='same')
                x = tf.layers.conv3d(x, 18, [1, 1, 5], activation=tf.nn.relu, padding='same')

                x = tf.layers.conv3d(x, 18, [3, 3, 1], activation=tf.nn.relu, padding='same')
                x = tf.layers.conv3d(x, 18, [1, 1, 5], activation=tf.nn.relu, padding='same')

                x = tf.layers.conv3d(x, 18, [3, 3, 1], activation=tf.nn.relu, padding='same')
                x = tf.layers.conv3d(x, 18, [1, 1, 5], activation=tf.nn.relu, padding='same')

                x = tf.layers.max_pooling3d(x, [2, 2, 2], strides=2) # 10, 10, 12

                x = tf.layers.conv3d(x, 18, [3, 3, 1], activation=tf.nn.relu, padding='same')
                x = tf.layers.conv3d(x, 18, [1, 1, 5], activation=tf.nn.relu, padding='same')

                x = tf.layers.max_pooling3d(x, [2, 2, 2], strides=2) # 5, 5, 6

                x = tf.layers.conv3d(x, 18, [3, 3, 1], activation=tf.nn.relu, padding='same')
                x = tf.layers.conv3d(x, 18, [1, 1, 5], activation=tf.nn.relu, padding='same')

                x = tf.layers.max_pooling3d(x, [2, 2, 2], strides=2) # 2, 2, 3

                x = tf.layers.conv3d(x, 18, [2, 2, 1], activation=tf.nn.relu, padding='same')
                x = tf.layers.conv3d(x, 18, [1, 1, 3], activation=tf.nn.relu, padding='same')

                flattened_features = tf.reshape(x, (self.batch_size, -1))

                fc_1 = tf.layers.dense(flattened_features, units=30, activation=tf.nn.relu)
                fc_2 = tf.layers.dense(fc_1, units=30, activation=tf.nn.relu)
                fc_3 = tf.layers.dense(fc_2, units=self.num_classes, activation=None)

                return fc_3

    def get_variable_scope(self):
        return 'h3d_conv_energy'

    def get_human_name(self):
        return 'Conv 3d Energy'