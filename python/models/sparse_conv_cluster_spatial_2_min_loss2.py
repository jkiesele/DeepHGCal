import tensorflow as tf
from models.sparse_conv_clustering_base import SparseConvClusteringBase
from ops.sparse_conv import *
from models.switch_model import SwitchModel


class SparseConvClusteringSpatialMinLoss2(SparseConvClusteringBase):

    def __init__(self, n_space, n_space_local, n_others, n_target_dim, batch_size, max_entries, learning_rate=0.0001):
        super(SparseConvClusteringSpatialMinLoss2, self).__init__(n_space, n_space_local, n_others, n_target_dim, batch_size, max_entries,
                                          learning_rate)
        self.weight_weights = []
        self.AdMat = None
        self.use_seeds = True
        
        

    def make_placeholders(self):
        self._placeholder_space_features = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries+1, self.n_space])
        self._placeholder_space_features_local = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries+1, self.n_space_local])
        self._placeholder_other_features = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries+1, self.n_other_features])
        self._placeholder_targets = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries+1, self.n_target_dim])
        self._placeholder_num_entries = tf.placeholder(dtype=tf.int64, shape=[self.batch_size, 1])


    def _get_loss(self):
        assert self._graph_output.shape[2] == 2

        num_entries = tf.squeeze(self._placeholder_num_entries, axis=1)
        print('num_entries',num_entries.shape)
        energy = self._placeholder_other_features[:, :, 0]
        
        prediction = self._graph_output
        targets = self._placeholder_targets
        
        maxlen = self.max_entries
        if self.use_seeds:
            energy=energy[:,0:-1] 
            targets = targets[:,0:-1,:]
            
            
            
        print('prediction',prediction.shape)
        print('self._placeholder_targets',self._placeholder_targets.shape)
        print('prediction',prediction.shape)
        print('targets',targets.shape)
        print('maxlen',maxlen)
        print('energy',energy.shape)

        diff_sq_1 = (prediction[:,:,0:2] - targets) ** 2 * tf.cast(
            tf.sequence_mask(num_entries, maxlen=maxlen)[:, :,
            tf.newaxis], tf.float32) * energy[:,:, tf.newaxis]
        diff_sq_1 = tf.reduce_sum(diff_sq_1, axis=[-1, -2]) / tf.reduce_sum(energy, axis=-1)
        loss_unreduced_1 = (diff_sq_1 / tf.cast(num_entries, tf.float32)) * tf.cast(
            num_entries != 0, tf.float32)

        diff_sq_2 = (prediction[:,:,0:2] - (1-targets)) ** 2 * tf.cast(
            tf.sequence_mask(num_entries, maxlen=maxlen)[:, :,
            tf.newaxis], tf.float32) * energy[:,:, tf.newaxis]
        diff_sq_2 = tf.reduce_sum(diff_sq_2, axis=[-1, -2]) / tf.reduce_sum(energy, axis=-1)
        loss_unreduced_2 = (diff_sq_2 / tf.cast(num_entries, tf.float32)) * tf.cast(
            num_entries != 0, tf.float32)

        shower_indices = tf.argmin(tf.concat((loss_unreduced_1[:, tf.newaxis], loss_unreduced_2[:, tf.newaxis]), axis=-1), axis=-1)

        condition_1 = tf.to_float(tf.equal((tf.to_float(shower_indices)[:, tf.newaxis, tf.newaxis]), 0.0))
        condition_2 = tf.to_float(tf.equal((tf.to_float(shower_indices)[:, tf.newaxis, tf.newaxis]), 1.0))
        sorted_target = targets * condition_1 + (1-targets) * condition_2

        # + (1-targets) * tf.cast(shower_indices[:,tf.newaxis,tf.newaxis]==1, tf.float32)

        perf1 = tf.reduce_sum(prediction[:,:,0] * energy, axis=[-1]) / tf.reduce_sum(sorted_target[:,:,0] * energy, axis=[-1])
        perf2 = tf.reduce_sum(prediction[:,:,1] * energy, axis=[-1]) / tf.reduce_sum(sorted_target[:,:,1] * energy, axis=[-1])


        self.mean_resolution, self.variance_resolution = tf.nn.moments(tf.concat((perf1, perf2), axis=0), axes=0)

        #return tf.reduce_mean(loss_unreduced_1)
        return tf.reduce_mean(tf.minimum(loss_unreduced_1, loss_unreduced_2))


    def _compute_output(self):
        # # nl_all = tf.layers.dense(tf.scalar_mul(0.001, self._placeholder_all_features), units=8, activation=tf.nn.relu)
        # # nl_all = tf.layers.dense(nl_all, units=8, activation=tf.nn.relu)
        # # nl_all = tf.layers.dense(nl_all, units=8, activation=tf.nn.relu)
        #
        # # TODO: Remove it later after regenerating the data, this only picks energy (or do something similar)
        # net = self._placeholder_all_features
        # net = tf.concat((net, self._placeholder_space_features_local), axis=2)

        # TODO: Will cause problems with batch size of 1
        
        feat = self._placeholder_other_features
        print("feat",feat.shape)
        space_feat = self._placeholder_space_features
        local_space_feat = self._placeholder_space_features_local
        num_entries = self._placeholder_num_entries
        
        seed_idxs=None
        
        if self.use_seeds:
            feat=feat[:,0:-1,:]
            space_feat=space_feat[:,0:-1,:]
            local_space_feat=local_space_feat[:,0:-1,:]
            #num_entries=num_entries[:,0:-1,:]
            idxa=tf.expand_dims(feat[:,-1,0], axis=1)
            idxb=tf.expand_dims(space_feat[:,-1,0], axis=1)
            seed_idxs=tf.concat([idxa, idxb], axis=-1)
            seed_idxs=tf.cast(seed_idxs+0.1,dtype=tf.int32)
            print(seed_idxs.shape)
            
        _input = construct_sparse_io_dict(feat, space_feat, local_space_feat,
                                          tf.squeeze(num_entries))
        net = _input
        
        net = sparse_conv_seeded(net,seed_idxs,nfilters=16,nspacefilters=16, nspacetransform=12)
        net = sparse_conv_seeded(net,seed_idxs,nfilters=16,nspacefilters=8, nspacetransform=12)
        net = sparse_conv_seeded(net,seed_idxs,nfilters=1,nspacefilters=1, nspacetransform=1,add_to_orig=False)
        



        
        #net,self.AdMat = sparse_conv_full_adjecency(_input, [64,32,16,4], AdMat=self.AdMat)
        #net,_ = sparse_conv_full_adjecency(net, [64,32,4], AdMat=self.AdMat)
        #net,_ = sparse_conv_full_adjecency(net, [64,32,2], AdMat=self.AdMat)
        #net,AdMat = sparse_conv_full_adjecency(net, 4, AdMat=AdMat)
        #net,AdMat = sparse_conv_full_adjecency(net, 4, AdMat=AdMat)
        #net,AdMat = sparse_conv_full_adjecency(net, 3, AdMat=AdMat, iterations=1)
        #just multple multiplications
        #net,AdMat = sparse_conv_full_adjecency(net, 3, AdMat=AdMat)
        #net,_ = sparse_conv_full_adjecency(net, 3, AdMat=AdMat)
        #net = sparse_conv_make_neighbors(_input, num_neighbors=18, output_all=3, n_transformed_spatial_features=3, propagrate_ahead=True)

      
        output = net['all_features'] # * tf.cast(tf.sequence_mask(tf.squeeze(self._placeholder_num_entries, axis=1), maxlen=self.max_entries)[:,:,tf.newaxis], tf.float32)
        output = tf.nn.softmax(output)

        self._graph_temp = output

        return output

    def get_variable_scope(self):
        return 'sparse_conv_clustering_spatial1'


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
            self._graph_summary_loss = tf.summary.scalar('loss', self._graph_loss)
            self._graph_summaries = tf.summary.merge([self._graph_summary_loss, tf.summary.scalar('mean-res', self.mean_resolution), tf.summary.scalar('variance-res', self.variance_resolution)])

            self._graph_summary_loss_validation = tf.summary.scalar('Validation Loss', self._graph_loss)
            self._graph_summaries_validation = tf.summary.merge([self._graph_summary_loss_validation])

    def get_losses(self):
        print("Hello, world!")
        return self._graph_loss