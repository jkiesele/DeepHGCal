import tensorflow as tf
from models.model import Model
from ops.sparse_conv import sparse_conv
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
        self.__construct_graphs()

    def get_summary(self):
        return self.__graph_summaries

    def get_summary_validation(self):
        return self.__graph_summaries_validation

    def get_placeholders(self):
        return self.__placeholder_space_features, self.__placeholder_all_features, self.__placeholder_neighbors_matrix,\
               self.__placeholder_labels, self.__placeholder_num_entries

    def get_compute_graphs(self):
        return self.__graph_logits, self.__graph_prediction

    def get_losses(self):
        return self.__graph_loss

    def get_optimizer(self):
        return self.__graph_optimizer

    def get_accuracy(self):
        return self.__accuracy

    def __construct_graphs(self):
        self.initialized = True

        self.__placeholder_space_features = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries, self.n_space])
        self.__placeholder_all_features = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries, self.n_all])
        self.__placeholder_neighbors_matrix = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_entries, self.n_max_neighbors])
        self.__placeholder_labels = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.num_classes])
        self.__placeholder_num_entries = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
        #
        # layer_1_space, layer_1_all = sparse_conv(self.__placeholder_space_features, self.__placeholder_all_features,
        #                                          self.__placeholder_neighbors_matrix)
        #
        # layer_2_space, layer_2_all = sparse_conv(layer_1_space, layer_1_all,
        #                                          self.__placeholder_neighbors_matrix)
        #
        # layer_3_space, layer_3_all = sparse_conv(layer_2_space, layer_2_all,
        #                                          self.__placeholder_neighbors_matrix)

        
        
        layer_1_out, layer_1_out_spatial = sparse_conv_2(self.__placeholder_space_features, self.__placeholder_all_features, self.__placeholder_neighbors_matrix, 15)
        layer_1_out = printLayerStuff(layer_1_out, "layer_1_out")
        layer_1_out_spatial = printLayerStuff(layer_1_out_spatial, "layer_1_out_spatial")
        
        layer_2_out, layer_2_out_spatial = sparse_conv_2(layer_1_out_spatial, layer_1_out, self.__placeholder_neighbors_matrix, 20)
        layer_3_out, layer_3_out_spatial = sparse_conv_2(layer_2_out_spatial, layer_2_out, self.__placeholder_neighbors_matrix, 25)
        layer_4_out, layer_4_out_spatial = sparse_conv_2(layer_3_out_spatial, layer_3_out, self.__placeholder_neighbors_matrix, 30)
        layer_4_out = printLayerStuff(layer_4_out, "layer_4_out")
        layer_4_out_spatial = printLayerStuff(layer_4_out_spatial, "layer_4_out_spatial")
        
        layer_5_out, layer_5_out_spatial = sparse_conv_2(layer_4_out_spatial, layer_4_out, self.__placeholder_neighbors_matrix, 35)
        layer_6_out, layer_6_out_spatial = sparse_conv_2(layer_5_out_spatial, layer_5_out, self.__placeholder_neighbors_matrix, 40)

        
        
        #layer_6_out_spatial=tf.Print(layer_6_out_spatial,[layer_6_out_spatial], "layer_6_out_spatial is ",summarize=2000)
        # TODO: Verify this code
        mask = tf.cast(tf.expand_dims(tf.sequence_mask(self.__placeholder_num_entries, maxlen=self.max_entries), axis=2), tf.float32)

        print(mask.shape)

        # mask=printLayerStuff(mask)
        
        nonzeros = tf.count_nonzero(mask, axis=1, dtype=tf.float32)
        
        
        
        print(nonzeros.shape)
        
        # jsut safety measure - should not be necessary once done
        flattened_features = tf.clip_by_value(tf.reduce_sum(layer_2_out * mask, axis=1) / nonzeros, 
                                              clip_value_min=-1e9, 
                                              clip_value_max=1e9)
        
        #flattened_features = tf.Print(flattened_features, [flattened_features], "flattened_features ")
        
        print(flattened_features.shape)
        # flattened_features=printLayerStuff(flattened_features)

        fc_1 = tf.layers.dense(flattened_features, units=100, activation=tf.nn.relu)
        fc_2 = tf.layers.dense(fc_1, units=100, activation=tf.nn.relu)
        fc_3 = tf.layers.dense(fc_2, units=self.num_classes, activation=None)

        self.__graph_logits = fc_3
        self.__graph_prediction = tf.argmax(self.__graph_logits, axis=1)
        
        self.__accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.__placeholder_labels, axis=1), self.__graph_prediction), tf.float32)) * 100
        self.__graph_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc_3, labels=self.__placeholder_labels))

        self.__graph_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.__graph_loss)

        # Repeating, maybe there is a better way?
        self.__graph_summary_loss = tf.summary.scalar('Loss', self.__graph_loss)
        self.__graph_summary_accuracy = tf.summary.scalar('Accuracy', self.__accuracy)
        self.__graph_summaries = tf.summary.merge([self.__graph_summary_loss, self.__graph_summary_accuracy])

        self.__graph_summary_loss_validation = tf.summary.scalar('Validation Loss', self.__graph_loss)
        self.__graph_summary_accuracy_validation = tf.summary.scalar('Validation Accuracy', self.__accuracy)
        self.__graph_summaries_validation = tf.summary.merge([self.__graph_summary_loss, self.__graph_summary_accuracy])


        #
        # # Dummy from now on
        # print("Whatever", ultimate_output.get_shape().as_list())
        # init = tf.global_variables_initializer()
        # with tf.Session() as sess:
        #     sess.run(init)
        #     while True:
        #         loss,_ = sess.run([self.__graph_loss, self.__graph_optimizer], feed_dict={
        #             self.__placeholder_space_features : np.zeros([self.batch_size, self.max_entries, self.n_space]),
        #             self.__placeholder_all_features: np.zeros([self.batch_size, self.max_entries, self.n_all]),
        #             self.__placeholder_neighbors_matrix: np.zeros([self.batch_size, self.max_entries, self.n_max_neighbors], dtype=np.int32),
        #             self.__placeholder_dummy_output: np.zeros([self.batch_size, self.max_entries, 1])+1000
        #         })
        #         print(loss)
