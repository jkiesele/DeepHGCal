import tensorflow as tf
from models.sparse_conv_clustering_base import SparseConvClusteringBase
from ops.sparse_conv import *
from models.switch_model import SwitchModel


class SparseConvClusteringMakeNeighborsNew(SparseConvClusteringBase):

    def __init__(self, n_space, n_space_local, n_others, n_target_dim, batch_size, max_entries, learning_rate=0.0001):
        super(SparseConvClusteringMakeNeighborsNew, self).__init__(n_space, n_space_local, n_others,
                                                                   n_target_dim, batch_size, max_entries,
                                                                   learning_rate)
        self.weight_weights = []
        self.AdMat = None
        self.use_seeds = True

        self.fixed_seeds = None

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
        assert self._graph_output.shape[2] == 3

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

        diff_sq_1 = (prediction[:, :, 0:2] - targets) ** 2 * tf.cast(
            tf.sequence_mask(num_entries, maxlen=self.max_entries)[:, :,
            tf.newaxis], tf.float32) * sqrt_energy[:, :, tf.newaxis]
        diff_sq_1 = tf.reduce_sum(diff_sq_1, axis=[-1, -2]) / tf.reduce_sum(sqrt_energy, axis=-1)
        loss_unreduced_1 = (diff_sq_1 / tf.cast(num_entries, tf.float32)) * tf.cast(
            num_entries != 0, tf.float32)

        diff_sq_2 = (prediction[:, :, 0:2] - (1 - targets)) ** 2 * tf.cast(
            tf.sequence_mask(num_entries, maxlen=self.max_entries)[:, :,
            tf.newaxis], tf.float32) * sqrt_energy[:, :, tf.newaxis]
        diff_sq_2 = tf.reduce_sum(diff_sq_2, axis=[-1, -2]) / tf.reduce_sum(sqrt_energy, axis=-1)
        loss_unreduced_2 = (diff_sq_2 / tf.cast(num_entries, tf.float32)) * tf.cast(
            num_entries != 0, tf.float32)

        shower_indices = tf.argmin(
            tf.concat((loss_unreduced_1[:, tf.newaxis], loss_unreduced_2[:, tf.newaxis]), axis=-1), axis=-1)

        condition_1 = tf.to_float(tf.equal((tf.to_float(shower_indices)[:, tf.newaxis, tf.newaxis]), 0.0))
        condition_2 = tf.to_float(tf.equal((tf.to_float(shower_indices)[:, tf.newaxis, tf.newaxis]), 1.0))
        sorted_target = targets * condition_1 + (1 - targets) * condition_2

        # + (1-targets) * tf.cast(shower_indices[:,tf.newaxis,tf.newaxis]==1, tf.float32)

        perf1 = tf.reduce_sum(prediction[:, :, 0] * energy, axis=[-1]) / tf.reduce_sum(sorted_target[:, :, 0] * energy,
                                                                                       axis=[-1])
        perf2 = tf.reduce_sum(prediction[:, :, 1] * energy, axis=[-1]) / tf.reduce_sum(sorted_target[:, :, 1] * energy,
                                                                                       axis=[-1])

        mean_resolution, variance_resolution = tf.nn.moments(tf.concat((perf1, perf2), axis=0), axes=0)

        self.mean_resolution = tf.clip_by_value(mean_resolution, 0.2, 2)
        self.variance_resolution = tf.clip_by_value(variance_resolution, 0, 1) / tf.clip_by_value(mean_resolution, 0.2,
                                                                                                  2)

        return tf.reduce_mean(loss_unreduced_1) * 1000.
        return tf.reduce_mean(tf.minimum(loss_unreduced_1, loss_unreduced_2)) * 1000.

    def _get_loss(self):
        return self.get_loss2()

    def compute_output_neighbours(self, _input, seeds):
        momentum = 0.9

        propagrate_ahead = True

        net = _input
        net = sparse_conv_add_simple_seed_labels(net, seeds)
        net = sparse_conv_batchnorm(net, momentum=momentum)
        out = []
        neta, netb = sparse_conv_split_batch(net, int(float(int(net['all_features'].shape[0])) / 2))
        nets = [neta, netb]

        # for i in range(len(nets)):
        #    with tf.device("/job:localhost/replica:0/task:0/device:GPU:"+str(i)):
        #        with tf.variable_scope('netscope',reuse=tf.AUTO_REUSE) as vscope:
        #            net = nets[i]
        net = sparse_conv_mix_colours_to_space(net)
        net = sparse_conv_make_neighbors2(net, num_neighbors=16, output_all=[16 for i in range(16)],
                                          space_transformations=[16, 4],
                                          propagrate_ahead=propagrate_ahead,
                                          strict_global_space=False,
                                          name="1")
        net = sparse_conv_batchnorm(net, momentum=momentum)

        net = sparse_conv_mix_colours_to_space(net)
        net = sparse_conv_make_neighbors2(net, num_neighbors=8, output_all=[16 for i in range(20)],
                                          space_transformations=[16, 4],
                                          propagrate_ahead=propagrate_ahead,
                                          strict_global_space=False,
                                          name="2")
        net = sparse_conv_batchnorm(net, momentum=momentum)

        flatout = sparse_conv_collapse(net)
        flatout = tf.layers.dense(flatout, 3, activation=tf.nn.relu)
        flatout = tf.nn.softmax(flatout)
        out.append(flatout)

        output = tf.concat(out, axis=0)
        return output

    def _compute_output(self):
        feat = self._placeholder_other_features
        print("feat", feat.shape)
        space_feat = self._placeholder_space_features
        local_space_feat = self._placeholder_space_features_local
        num_entries = self._placeholder_num_entries
        n_batch = space_feat.shape[0]

        seed_idxs = None

        seeds = self._placeholder_seed_indices
        seeds = tf.transpose(seeds, [1, 0])
        seeds = tf.random_shuffle(seeds)
        seeds = tf.transpose(seeds, [1, 0])
        # seeds = self._placeholder_seed_indices
        print('seeds', seeds.shape)

        _input = construct_sparse_io_dict(feat, space_feat, local_space_feat,
                                          tf.squeeze(num_entries))

        simple_input = tf.concat([space_feat, local_space_feat, feat], axis=-1)
        output = self.compute_output_neighbours(_input, seeds)  # self._placeholder_seed_indices)
        self._graph_temp = tf.reduce_sum(output[:, :, :], axis=1) / 2679.

        return output

    def get_variable_scope(self):
        return 'sparse_conv_clustering_spatial1'

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
            self._graph_summaries = tf.summary.merge(
                [self._graph_summary_loss, tf.summary.scalar('mean-res', self.mean_resolution),
                 tf.summary.scalar('variance-res', self.variance_resolution)])

            self._graph_summary_loss_validation = tf.summary.scalar('Validation Loss', self._graph_loss)
            self._graph_summaries_validation = tf.summary.merge([self._graph_summary_loss_validation])

    def get_losses(self):
        print("Hello, world!")
        return self._graph_loss