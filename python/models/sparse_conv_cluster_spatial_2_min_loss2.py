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
        
        self.fixed_seeds=None

    def make_placeholders(self):
        self._placeholder_space_features = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries, self.n_space])
        self._placeholder_space_features_local = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries, self.n_space_local])
        self._placeholder_other_features = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries, self.n_other_features])
        self._placeholder_targets = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries, self.n_target_dim])
        self._placeholder_num_entries = tf.placeholder(dtype=tf.int64, shape=[self.batch_size, 1])
        self._placeholder_seed_indices = tf.placeholder(dtype=tf.int64, shape=[self.batch_size, 2])

    def get_placeholders(self):
        return self._placeholder_space_features,self._placeholder_space_features_local, self._placeholder_other_features, \
               self._placeholder_targets, self._placeholder_num_entries, self._placeholder_seed_indices


    def get_loss2(self):
        assert self._graph_output.shape[2] == 3

        num_entries = tf.squeeze(self._placeholder_num_entries, axis=1)
        print('num_entries',num_entries.shape)
        energy = self._placeholder_other_features[:, :, 0]
        
        prediction = self._graph_output
        targets = self._placeholder_targets
        
        maxlen = self.max_entries
        #if self.use_seeds:
        #    energy=energy[:,0:-1] 
        #    targets = targets[:,0:-1,:]

        diff_sq_1 = (prediction[:,:,0:2] - targets) ** 2 * tf.cast(
            tf.sequence_mask(num_entries, maxlen=self.max_entries)[:, :,
            tf.newaxis], tf.float32) * energy[:,:, tf.newaxis]
        diff_sq_1 = tf.reduce_sum(diff_sq_1, axis=[-1, -2]) / tf.reduce_sum(energy, axis=-1)
        loss_unreduced_1 = (diff_sq_1 / tf.cast(num_entries, tf.float32)) * tf.cast(
            num_entries != 0, tf.float32)

        diff_sq_2 = (prediction[:,:,0:2] - (1-targets)) ** 2 * tf.cast(
            tf.sequence_mask(num_entries, maxlen=self.max_entries)[:, :,
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

        mean_resolution, variance_resolution = tf.nn.moments(tf.concat((perf1, perf2), axis=0), axes=0)


        self.mean_resolution = tf.clip_by_value(mean_resolution, -2, 3)
        self.variance_resolution = tf.clip_by_value(variance_resolution, 0, 5)/tf.clip_by_value(mean_resolution,0.1,3)

        #return tf.reduce_mean(loss_unreduced_1)*1000.
        return tf.reduce_mean(tf.minimum(loss_unreduced_1, loss_unreduced_2))*1000.

    def _get_loss(self):
        
        return self.get_loss2()
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
        
        true_energy_pc = targets * energy[:,:, tf.newaxis]
        predicted_fraction_pc = prediction[:,:,0:2]
        predicted_fraction_pc_swapped = tf.concat([tf.expand_dims(prediction[:,:,1],axis=2),
                                                 tf.expand_dims(prediction[:,:,0],axis=2)],
                                                 axis=-1)
                                                 
        true_shower_energies = tf.reduce_sum(true_energy_pc,axis=1)
        predicted_shower_energies =  tf.reduce_sum(predicted_fraction_pc * energy[:,:, tf.newaxis],axis=1)
        predicted_shower_energies_swapped =  tf.reduce_sum(predicted_fraction_pc_swapped * energy[:,:, tf.newaxis],axis=1)
        
        predicted_energy_fraction = predicted_fraction_pc * energy[:,:, tf.newaxis]
        predicted_energy_fraction_swapped =  predicted_fraction_pc_swapped * energy[:,:, tf.newaxis]
        
        resolution = tf.where(tf.less( tf.abs(predicted_shower_energies-true_shower_energies), tf.abs(predicted_shower_energies_swapped - true_shower_energies)),
                              predicted_shower_energies,
                              predicted_shower_energies_swapped)
        
        resolution = resolution[:,0]/true_shower_energies[:,0] -1 #one is enough
        
        print('resolution',resolution.shape)
        
        mean_resolution, variance_resolution = tf.nn.moments( resolution , axes=0)
        self.mean_resolution, self.variance_resolution = tf.clip_by_value(mean_resolution, -1, 1), tf.clip_by_value(variance_resolution,0,3)
        
        
        print('self.mean_resolution',self.mean_resolution.shape)
        print('self.mean_resolution',self.mean_resolution.shape)
        
        loss_direct = (predicted_energy_fraction-true_energy_pc)**2
        loss_swapped = (predicted_energy_fraction_swapped-true_energy_pc)**2
        
        loss_direct=tf.reduce_mean(tf.reduce_mean(loss_direct, axis=-1), axis=-1)
        loss_swapped=tf.reduce_mean(tf.reduce_mean(loss_swapped, axis=-1), axis=-1)
        
        print('loss_direct',loss_direct.shape)
        
        #return tf.reduce_mean(loss_direct)
        return tf.reduce_mean(tf.minimum(loss_direct, loss_swapped))
        
        
        
        #####
        
        # do cat cross entropy with prediction and targets and then weight by energy for both combinations
        loss_direct = tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets,logits=predicted_fraction_pc,dim=-1) # * energy / tf.reduce_sum(energy,keepdims=True)
        loss_swapped = tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets,logits=predicted_fraction_pc_swapped,dim=-1) # * energy / tf.reduce_sum(energy,keepdims=True)
        
        loss_direct = tf.reduce_mean(loss_direct, axis=-1)
        loss_swapped = tf.reduce_mean(loss_swapped, axis=-1)
        
        return tf.reduce_mean(loss_direct)
        
        return tf.reduce_mean(tf.minimum(loss_direct, loss_swapped))
    
    
        
        logits = predicted_energy_pc / tf.expand_dims(true_shower_energies, axis=1) 
        logits_swapped = predicted_energy_pc_swapped / tf.expand_dims(true_shower_energies, axis=1)
        
        
        print('true_shower_energies',true_shower_energies.shape)
        print('true_energy_pc',true_energy_pc.shape)
        print('diff2',diff2.shape)

        exit()


        diff_sq_1 = (prediction[:,:,0:2] - targets) ** 2 * energy[:,:, tf.newaxis] #* tf.cast(
            #tf.sequence_mask(num_entries, maxlen=maxlen)[:, :,
            #tf.newaxis], tf.float32) 
        
        print('energy[:,:, tf.newaxis]',energy[:,:, tf.newaxis])
        totalE = energy[:,:, tf.newaxis]*targets
        totalE = tf.reduce_sum(totalE,axis=1)
        totalE = tf.expand_dims(totalE, axis=1)
        print('totalE',totalE)
        
        diff_sq_1=diff_sq_1/totalE
        lossout= tf.reduce_mean(diff_sq_1);
        
        
        print('lossout',lossout)
        
        #exit()
        
        mean_resolution, variance_resolution = tf.nn.moments(diff_sq_1, axes=0)
        
        self.mean_resolution, self.variance_resolution = tf.clip_by_value(mean_resolution, -2, 2), tf.clip_by_value(variance_resolution,0,10)
        
        return lossout
    
        
        #exit()
        
        diff_sq_1 = tf.reduce_sum(diff_sq_1, axis=[-1, -2]) #/ tf.reduce_sum(energy, axis=-1)
        loss_unreduced_1 = diff_sq_1

        diff_sq_2 = (prediction[:,:,0:2] - (1-targets)) ** 2 * energy[:,:, tf.newaxis] #* tf.cast(
            #tf.sequence_mask(num_entries, maxlen=maxlen)[:, :,
            #tf.newaxis], tf.float32) 
        diff_sq_2 = tf.reduce_sum(diff_sq_2, axis=[-1, -2]) #/ tf.reduce_sum(energy, axis=-1)
        loss_unreduced_2 = diff_sq_2

        shower_indices = tf.argmin(tf.concat((loss_unreduced_1[:, tf.newaxis], loss_unreduced_2[:, tf.newaxis]), axis=-1), axis=-1)

        condition_1 = tf.to_float(tf.equal((tf.to_float(shower_indices)[:, tf.newaxis, tf.newaxis]), 0.0))
        condition_2 = tf.to_float(tf.equal((tf.to_float(shower_indices)[:, tf.newaxis, tf.newaxis]), 1.0))
        sorted_target = targets * condition_1 + (1-targets) * condition_2

        # + (1-targets) * tf.cast(shower_indices[:,tf.newaxis,tf.newaxis]==1, tf.float32)

        perf1 = tf.reduce_sum(prediction[:,:,0] * energy, axis=[-1]) / tf.reduce_sum(sorted_target[:,:,0] * energy, axis=[-1])
        perf2 = tf.reduce_sum(prediction[:,:,1] * energy, axis=[-1]) / tf.reduce_sum(sorted_target[:,:,1] * energy, axis=[-1])


        self.mean_resolution, self.variance_resolution = tf.nn.moments(tf.concat((perf1, perf2), axis=0), axes=0)

        
        #return tf.reduce_mean(loss_unreduced_1)
        #this is the symmetrised
        return tf.reduce_mean(tf.minimum(loss_unreduced_1, loss_unreduced_2))

    def compute_output_seed_driven(self,_input,seed_idxs):
        
        momentum=0.9
        
        
        _input = sparse_conv_batchnorm(_input,momentum=momentum)
       
        net = _input
        net = sparse_conv_seeded(net,seed_idxs,nfilters=4, nspacefilters=128, nspacetransform=1,nspacedim=3)#,original_dict=_input)
        net = sparse_conv_batchnorm(net,momentum=momentum)
        #net = sparse_conv_make_neighbors(net, num_neighbors=6, output_all=4, spatial_degree_non_linearity=1,propagrate_ahead=False)
        
        net = sparse_conv_seeded(net,seed_idxs,nfilters=16,nspacefilters=128, nspacetransform=1,nspacedim=3)#original_dict=_input)
        net = sparse_conv_batchnorm(net,momentum=momentum)
        net = sparse_conv_seeded(net,seed_idxs,nfilters=32,nspacefilters=128, nspacetransform=1,nspacedim=3)#original_dict=_input)
        net = sparse_conv_batchnorm(net,momentum=momentum)
        net = sparse_conv_seeded(net,seed_idxs,nfilters=64,nspacefilters=128, nspacetransform=1,nspacedim=3)#original_dict=_input)
        net = sparse_conv_batchnorm(net,momentum=momentum)
        net = sparse_conv_seeded(net,seed_idxs,nfilters=64,nspacefilters=128, nspacetransform=1,nspacedim=3)#original_dict=_input)
        net = sparse_conv_batchnorm(net,momentum=momentum)
        net = sparse_conv_seeded(net,seed_idxs,nfilters=64,nspacefilters=64, nspacetransform=1,nspacedim=3)#original_dict=_input)
        net = sparse_conv_batchnorm(net,momentum=momentum)
        net = sparse_conv_seeded(net,seed_idxs,nfilters=32,nspacefilters=64, nspacetransform=1,nspacedim=3)#original_dict=_input)
        net = sparse_conv_batchnorm(net,momentum=momentum)
        net = sparse_conv_seeded(net,seed_idxs,nfilters=32,nspacefilters=64, nspacetransform=1,nspacedim=3)#original_dict=_input)
        
        net = sparse_conv_seeded(net,seed_idxs,nfilters=16,nspacefilters=1, nspacetransform=1,nspacedim=3)#,add_to_orig=False)

        output = net['all_features'] # * tf.cast(tf.sequence_mask(tf.squeeze(self._placeholder_num_entries, axis=1), maxlen=self.max_entries)[:,:,tf.newaxis], tf.float32)
        output = tf.layers.dense(output,3,activation=tf.nn.relu)
        output = tf.nn.softmax(output)
        return output
        
    def compute_output_full_adjecency(self,_input):
        
        momentum=0.1
        _input = sparse_conv_batchnorm(_input,momentum=momentum)
        net=_input
        net,AdMat = sparse_conv_full_adjecency(net,nfilters=[128,64,64,16,4,2],noutputfilters=-32, AdMat=None,  iterations=1,spacetransform=64)
        net       = sparse_conv_batchnorm(net,momentum=momentum)
        net,AdMat = sparse_conv_full_adjecency(net,nfilters=[128,64,64,16,4,2],noutputfilters=-32, AdMat=None, iterations=1,spacetransform=64)
        net       = sparse_conv_batchnorm(net,momentum=momentum)
        net,AdMat = sparse_conv_full_adjecency(net,nfilters=[128,64,64,16,4,2],noutputfilters=-32, AdMat=None, iterations=1,spacetransform=64)
        net       = sparse_conv_batchnorm(net,momentum=momentum)
        net,AdMat = sparse_conv_full_adjecency(net,nfilters=[128,64,64,16,4,2],noutputfilters=-32, AdMat=None, iterations=1,spacetransform=64)
        net       = sparse_conv_batchnorm(net,momentum=momentum)
        net,AdMat = sparse_conv_full_adjecency(net,nfilters=[128,64,64,16,4,2],noutputfilters=32, AdMat=None, iterations=1,spacetransform=64)
        net       = sparse_conv_batchnorm(net,momentum=momentum)
        
        output = net['all_features'] # * tf.cast(tf.sequence_mask(tf.squeeze(self._placeholder_num_entries, axis=1), maxlen=self.max_entries)[:,:,tf.newaxis], tf.float32)
        output = tf.layers.dense(output,3,activation=tf.nn.relu)
        net       = sparse_conv_batchnorm(net,momentum=momentum)
        output = tf.nn.softmax(output)
        return output
    
    def compute_output_seed_driven_neighbours(self,_input,seed_idxs):
        
        momentum=0.9
        
        propagate=True
        _input = sparse_conv_batchnorm(_input,momentum=momentum)
       
        net = _input
        net = sparse_conv_make_neighbors(net, num_neighbors=18, output_all=24, spatial_degree_non_linearity=3, propagrate_ahead=propagate)
        net = sparse_conv_seeded(net,seed_idxs,nfilters=16, nspacefilters=128, nspacetransform=1,nspacedim=3)#,original_dict=_input)
        net = sparse_conv_batchnorm(net,momentum=momentum)
        
        net = sparse_conv_make_neighbors(net, num_neighbors=18, output_all=24, spatial_degree_non_linearity=3, propagrate_ahead=propagate)
        net = sparse_conv_seeded(net,seed_idxs,nfilters=64,nspacefilters=128, nspacetransform=1,nspacedim=3)#original_dict=_input)
        net = sparse_conv_batchnorm(net,momentum=momentum)
        
        net = sparse_conv_make_neighbors(net, num_neighbors=18, output_all=24, spatial_degree_non_linearity=1, propagrate_ahead=propagate)
        net = sparse_conv_seeded(net,seed_idxs,nfilters=64,nspacefilters=128, nspacetransform=1,nspacedim=3)#original_dict=_input)
        net = sparse_conv_batchnorm(net,momentum=momentum)
        
        net = sparse_conv_make_neighbors(net, num_neighbors=18, output_all=22, spatial_degree_non_linearity=1, propagrate_ahead=propagate)
        net = sparse_conv_seeded(net,seed_idxs,nfilters=64,nspacefilters=100, nspacetransform=1,nspacedim=3)#original_dict=_input)
        net = sparse_conv_batchnorm(net,momentum=momentum)
        
        output = net['all_features'] # * tf.cast(tf.sequence_mask(tf.squeeze(self._placeholder_num_entries, axis=1), maxlen=self.max_entries)[:,:,tf.newaxis], tf.float32)
        output = tf.layers.dense(output,3,activation=tf.nn.relu)
        output = tf.nn.softmax(output)
        return output
    
    
    def compute_output_neighbours(self,_input):
        momentum=0.1
        
        propagrate_ahead=True
        
        net=_input
        net = sparse_conv_batchnorm(net,momentum=momentum)
        
        #neta, netb = sparse_conv_split_batch(net,int(float(int(net['all_features'].shape[0]))/2))
        #nets=[neta,netb]
        out=[]
        #for i in range(len(nets)):
        #    with tf.device("/job:localhost/replica:0/task:0/device:GPU:"+str(i)):
        #        net = nets[i]
        net = sparse_conv_mix_colours_to_space(net)
        net = sparse_conv_make_neighbors2(net, num_neighbors=24, output_all=32, 
                                         space_transformations=[64,32,16,4], 
                                         propagrate_ahead=propagrate_ahead,
                                         strict_global_space=False)
        net = sparse_conv_batchnorm(net,momentum=momentum)
        
        net = sparse_conv_mix_colours_to_space(net)
        net = sparse_conv_make_neighbors2(net, num_neighbors=24, output_all=32, 
                                         space_transformations=[64,32,16,4], 
                                         propagrate_ahead=propagrate_ahead,
                                         strict_global_space=False)
        net = sparse_conv_batchnorm(net,momentum=momentum)
        
        net = sparse_conv_mix_colours_to_space(net)
        net = sparse_conv_make_neighbors2(net, num_neighbors=24, output_all=32, 
                                         space_transformations=[64,32,16,4], 
                                         propagrate_ahead=propagrate_ahead,
                                         strict_global_space=False)
        net = sparse_conv_batchnorm(net,momentum=momentum)
        out.append(sparse_conv_collapse(net))
            
        

        output = tf.concat(out,axis=0)
        output = tf.layers.dense(output,3,activation=tf.nn.relu)
        output = tf.nn.softmax(output)
        return output
        
    def _compute_output(self):
        
        
        feat = self._placeholder_other_features
        print("feat",feat.shape)
        space_feat = self._placeholder_space_features
        local_space_feat = self._placeholder_space_features_local
        num_entries = self._placeholder_num_entries
        
        seed_idxs=None
        
        #if self.use_seeds:
        #    feat=feat[:,0:-1,:]
        #    space_feat=space_feat[:,0:-1,:]
        #    local_space_feat=local_space_feat[:,0:-1,:]
        #    #num_entries=num_entries[:,0:-1,:]
        #    idxa=tf.expand_dims(feat[:,-1,0], axis=1)
        #    idxb=tf.expand_dims(space_feat[:,-1,0], axis=1)
        #    seed_idxs=tf.concat([idxa, idxb], axis=-1)
        #    seed_idxs=tf.cast(seed_idxs+0.1,dtype=tf.int32)
        #    print(seed_idxs.shape)
            
        if self.fixed_seeds is None:
            self.fixed_seeds = tf.constant([i*2679 for i in range(0,11)],dtype=tf.int32)
            self.fixed_seeds = tf.expand_dims(self.fixed_seeds, axis=0)
            self.fixed_seeds = tf.tile(self.fixed_seeds,[space_feat.shape[0],1])
            print('self.fixed_seeds',self.fixed_seeds.shape)
        
        _input = construct_sparse_io_dict(feat, space_feat, local_space_feat,
                                          tf.squeeze(num_entries))
        
        simple_input = tf.concat([space_feat,local_space_feat,feat],axis=-1)
        #output=self.compute_output_seed_driven(_input,seed_idxs)
        #output = self.compute_output_seed_driven_neighbours(_input,self._placeholder_seed_indices)
        output = self.compute_output_neighbours(_input)
        #output=self.compute_output_seed_driven(_input,self._placeholder_seed_indices)
        #output=self.compute_output_full_adjecency(_input)
        
        self._graph_temp = tf.reduce_sum(output[:,:,0], axis=1)/2679.

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