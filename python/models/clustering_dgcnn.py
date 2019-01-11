import tensorflow as tf
from models.sparse_conv_clustering_base import SparseConvClusteringBase
# from ops.sparse_conv import *
from ops.sparse_conv_2 import *
from models.switch_model import SwitchModel
from ops.activations import *


class DynamicGraphCnnAlpha(SparseConvClusteringBase):

    def __init__(self, n_space, n_space_local, n_others, n_target_dim, batch_size, max_entries, learning_rate=0.0001):
        super(DynamicGraphCnnAlpha, self).__init__(n_space, n_space_local, n_others, n_target_dim,
                                                                  batch_size, max_entries,
                                                                  learning_rate)
        self.weight_weights = []
        self.AdMat = None
        self.use_seeds = True
        self.mean_sqrt_resolution = None
        self.variance_sqrt_resolution = None
        self.total_loss = None
        self.fixed_seeds = None
        self.momentum = 0.6
        self.varscope = 'sparse_conv_clustering_spatial1'

    def normalise_response(self,total_response):
        mean, variance = tf.nn.moments(total_response, axes=0)
        return tf.clip_by_value(mean, 0.01, 5), tf.clip_by_value(variance, 0, 5)/tf.clip_by_value(mean,0.01,5)


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

    def get_loss2(self):
        assert self._graph_output.shape[2] >= 2

        num_entries = tf.squeeze(self._placeholder_num_entries, axis=1)
        print('num_entries', num_entries.shape)
        energy = self._placeholder_other_features[:, :, 0]
        sqrt_energy = tf.sqrt(energy)

        prediction = self._graph_output
        targets = self._placeholder_targets

        maxlen = self.max_entries
        # if self.use_seeds:
        #    energy=energy[:,0:-1]
        #    targets = targets[:,0:-1,:]

        total_energy = tf.reduce_sum(energy, axis=-1)

        print('total_energy', total_energy.shape)

        energies = targets * energy[:, :, tf.newaxis]
        energy_sums = tf.reduce_sum(energies, axis=1)
        energy_a = energy_sums[:, 0]
        energy_b = energy_sums[:, 1]

        print('energy_a', energy_a.shape)

        sqrt_energies = targets * sqrt_energy[:, :, tf.newaxis]

        print('sqrt_energies', sqrt_energies.shape)

        sqrt_energy_sum = tf.reduce_sum(sqrt_energies, axis=1)
        sqrt_energy_a = sqrt_energy_sum[:, 0]
        sqrt_energy_b = sqrt_energy_sum[:, 1]

        print('sqrt_energy_sum', sqrt_energy_sum.shape)

        diff_sq = (prediction[:, :, 0:2] - targets) ** 2.
        diff_sq_a = diff_sq[:, :, 0]
        diff_sq_b = diff_sq[:, :, 1]

        print('diff_sq_a', diff_sq_a.shape)

        loss_a = tf.reduce_sum(diff_sq_a * sqrt_energies[:, :, 0], axis=1) / (sqrt_energy_a)
        loss_b = tf.reduce_sum(diff_sq_b * sqrt_energies[:, :, 1], axis=1) / (sqrt_energy_b)

        old_loss = (tf.reduce_sum(diff_sq_a * sqrt_energies[:, :, 0], axis=1) + tf.reduce_sum(
            diff_sq_b * sqrt_energies[:, :, 1], axis=1)) / (sqrt_energy_a + sqrt_energy_b)

        print('loss_a', loss_a.shape)

        total_loss = loss_a + loss_b

        response_a = tf.reduce_sum(prediction[:, :, 0] * energy, axis=1) / energy_a
        response_b = tf.reduce_sum(prediction[:, :, 1] * energy, axis=1) / energy_b

        print('response_a', response_a.shape)

        total_response = tf.concat([response_a, response_b], axis=0)

        self.mean_resolution, self.variance_resolution = self.normalise_response(total_response)

        self.total_loss = tf.reduce_mean(total_loss)

        sqrt_response_a = tf.reduce_sum(prediction[:, :, 0] * sqrt_energies[:, :, 0], axis=1) / sqrt_energy_a
        sqrt_response_b = tf.reduce_sum(prediction[:, :, 1] * sqrt_energies[:, :, 1], axis=1) / sqrt_energy_b

        sqrt_total_response = tf.concat([sqrt_response_a, sqrt_response_b], axis=0)

        self.mean_sqrt_resolution, self.variance_sqrt_resolution = self.normalise_response(sqrt_total_response)

        return tf.reduce_mean(
            old_loss)  # + tf.reduce_mean(0.1*tf.abs(1-self.mean_resolution)+0.1*self.variance_resolution)

    def _get_loss(self):

        return self.get_loss2()

    def compute_output_dgcnn(self, _input, seeds, dropout=-1.):

        self.freeze_bn_after = None
        feat = sparse_conv_collapse(_input)

        feat = tf.layers.dense(feat, 16)  # global transform to 3D
        feat = tf.layers.batch_normalization(feat, training=self.is_train, momentum=self.momentum)

        feat = zero_out_by_energy(feat)
        feat = sparse_conv_edge_conv(feat, 40, [64, 64, 64])
        feat_g = sparse_conv_global_exchange(feat)

        feat = tf.layers.dense(tf.concat([feat, feat_g], axis=-1),
                               64, activation=tf.nn.relu)

        feat = tf.layers.batch_normalization(feat, training=self.is_train, momentum=self.momentum)
        if dropout > 0:
            feat = tf.layers.dropout(feat, rate=dropout, training=self.is_train)

        feat = zero_out_by_energy(feat)
        feat1 = sparse_conv_edge_conv(feat, 40, [64, 64, 64])
        feat1_g = sparse_conv_global_exchange(feat1)
        feat1 = tf.layers.dense(tf.concat([feat1, feat1_g], axis=-1),
                                64, activation=tf.nn.relu)
        feat1 = tf.layers.batch_normalization(feat1, training=self.is_train, momentum=self.momentum)
        if dropout > 0:
            feat1 = tf.layers.dropout(feat1, rate=dropout, training=self.is_train)

        feat = zero_out_by_energy(feat)
        feat2 = sparse_conv_edge_conv(feat1, 40, [64, 64, 64])
        feat2_g = sparse_conv_global_exchange(feat2)
        feat2 = tf.layers.dense(tf.concat([feat2, feat2_g], axis=-1),
                                64, activation=tf.nn.relu)
        feat2 = tf.layers.batch_normalization(feat2, training=self.is_train, momentum=self.momentum)
        if dropout > 0:
            feat2 = tf.layers.dropout(feat2, rate=dropout, training=self.is_train)

        feat = zero_out_by_energy(feat)
        feat3 = sparse_conv_edge_conv(feat2, 40, [64, 64, 64])
        feat3 = tf.layers.batch_normalization(feat3, training=self.is_train, momentum=self.momentum)
        if dropout > 0:
            feat3 = tf.layers.dropout(feat3, rate=dropout, training=self.is_train)

        # global_feat = tf.layers.dense(feat2,1024,activation=tf.nn.relu)
        # global_feat = max_pool_on_last_dimensions(global_feat, skip_first_features=0, n_output_vertices=1)
        # print('global_feat',global_feat.shape)
        # global_feat = tf.tile(global_feat,[1,feat.shape[1],1])
        # print('global_feat',global_feat.shape)

        feat = tf.concat([feat, feat1, feat2, feat_g, feat1_g, feat2_g, feat3], axis=-1)
        feat = tf.layers.dense(feat, 32, activation=tf.nn.relu)
        feat = tf.layers.dense(feat, 3, activation=tf.nn.relu)
        return feat

    def _compute_output(self):

        feat = self._placeholder_other_features
        print("feat", feat.shape)
        space_feat = self._placeholder_space_features
        local_space_feat = self._placeholder_space_features_local
        num_entries = self._placeholder_num_entries
        n_batch = space_feat.shape[0]

        seed_idxs = None

        # if self.use_seeds:
        #    feat=feat[:,0:-1,:]
        #    space_feat=space_feat[:,0:-1,:]
        #    local_space_feat=local_space_feat[:,0:-1,:]
        #    #num_entries=num_entries[:,0:-1,:]
        #    idxa=tf.expand_dims(feat[:,-1,0], axis=1)
        #    idxb=tf.expand_dims(space_feat[:,-1,0], axis=1)
        #    seed_idxs=tf.concat([idxa, idxb], axis=-1)
        #    seed_idxs=tf.cast(seed_idxs+0.1,dtype=tf.int32)
        #    print(seed_idxs.shape)

        nrandom = 1
        random_seeds = tf.random_uniform(shape=(int(n_batch), nrandom), minval=0, maxval=2679, dtype=tf.int64)
        print('random_seeds', random_seeds.shape)
        seeds = tf.concat([self._placeholder_seed_indices, random_seeds], axis=-1)
        seeds = tf.transpose(seeds, [1, 0])
        seeds = tf.random_shuffle(seeds)
        seeds = tf.transpose(seeds, [1, 0])
        seeds = self._placeholder_seed_indices
        print('seeds', seeds.shape)

        net = construct_sparse_io_dict(feat, space_feat, local_space_feat,
                                       tf.squeeze(num_entries))

        net = sparse_conv_normalise(net, log_energy=True)

        output = self.compute_output_dgcnn(net, self._placeholder_seed_indices)

        # output=self.compute_output_seed_driven(net,self._placeholder_seed_indices)
        # output=self.compute_output_full_adjecency(_input)
        output = tf.layers.dense(output, 2)
        output = tf.nn.softmax(output)

        self._graph_temp = tf.reduce_sum(output[:, :, :], axis=1) / 2679.

        return output

    def get_variable_scope(self):
        return self.config_name

    def _construct_graphs(self):
        with tf.variable_scope(self.get_variable_scope()):
            self.initialized = True
            self.weight_init_width = 1e-6

            self.make_placeholders()

            self._graph_output = self._compute_output()

            # self._graph_temp = tf.nn.softmax(self.__graph_logits)

            self._graph_loss = self._get_loss()

            self._graph_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self._graph_loss)

            # Repeating, maybe there is a better way?
            self._graph_summary_loss = tf.summary.scalar('loss', self._graph_loss)
            self._graph_summaries = tf.summary.merge([self._graph_summary_loss,
                                                      tf.summary.scalar('mean-res', self.mean_resolution),
                                                      tf.summary.scalar('variance-res', self.variance_resolution),
                                                      tf.summary.scalar('mean-res-sqrt', self.mean_sqrt_resolution),
                                                      tf.summary.scalar('variance-res-sqrt',
                                                                        self.variance_sqrt_resolution),
                                                      tf.summary.scalar('fraction-loss', self.total_loss)])

            self._graph_summary_loss_validation = tf.summary.scalar('Validation Loss', self._graph_loss)
            self._graph_summaries_validation = tf.summary.merge([self._graph_summary_loss_validation])

    def get_losses(self):
        print("Hello, world!")
        return self._graph_loss