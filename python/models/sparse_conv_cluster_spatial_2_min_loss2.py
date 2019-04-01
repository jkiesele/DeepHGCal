import tensorflow as tf
from models.sparse_conv_clustering_base import SparseConvClusteringBase
#from ops.sparse_conv import *
from ops.sparse_conv_2 import *
from models.switch_model import SwitchModel
from ops.activations import *

class SparseConvClusteringSpatialMinLoss2(SparseConvClusteringBase):

    def __init__(self, n_space, n_space_local, n_others, n_target_dim, batch_size, max_entries, learning_rate=0.0001):
        super(SparseConvClusteringSpatialMinLoss2, self).__init__(n_space, n_space_local, n_others, n_target_dim, batch_size, max_entries,
                                          learning_rate)
        self.weight_weights = []
        self.AdMat = None
        self.use_seeds = True
        self.mean_sqrt_resolution=None
        self.variance_sqrt_resolution=None
        self.total_loss = None
        self.fixed_seeds=None
        self.momentum = 0.6
        self.varscope='sparse_conv_clustering_spatial1'
        self.freeze_bn_after=None
        self.E_loss=False
        self.sum_loss=True
        self.log_loss=False
        self.xentr_loss=False
        
    def normalise_response(self,total_response):
        mean, variance = tf.nn.moments(total_response, axes=0)
        return tf.clip_by_value(mean, 0.01, 100), tf.clip_by_value(variance, 0, 100)/tf.clip_by_value(mean,0.001,100)

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

    def get_loss_xentropy(self):
        assert self._graph_output.shape[2] >= 2
        
        prediction = self._graph_output
        targets = self._placeholder_targets
        prediction = tf.clip_by_value(prediction, 1e-5, 1 - 1e-5)
        
        single_xentr = - (targets * tf.log(prediction) + (1 - targets)*tf.log(1-prediction))
        
        
        return self.create_loss_weight_by_energy(single_xentr)
        
        
    def create_loss_weight_by_energy(self, loss_per_vertex):
        
        num_entries = tf.squeeze(self._placeholder_num_entries, axis=1)
        print('num_entries',num_entries.shape)
        energy = self._placeholder_other_features[:, :, 0]
        
        sqrt_energy = None
        if self.log_loss:
            sqrt_energy = tf.log(energy+1)
        else:
            sqrt_energy = tf.sqrt(energy)
        
        prediction = self._graph_output
        targets = self._placeholder_targets
        
        maxlen = self.max_entries
        #if self.use_seeds:
        #    energy=energy[:,0:-1] 
        #    targets = targets[:,0:-1,:]
        
        total_energy   = tf.reduce_sum(energy, axis=-1)
        
        print('total_energy',total_energy.shape)
        
        energies = targets * energy[:,:, tf.newaxis]
        energy_sums = tf.reduce_sum(energies , axis=1)
        energy_a = energy_sums[:,0]
        energy_b = energy_sums[:,1]
        
        print('energy_a',energy_a.shape)
        
        sqrt_energies = targets * sqrt_energy[:,:, tf.newaxis]
        
        print('sqrt_energies',sqrt_energies.shape)
        
        sqrt_energy_sum = tf.reduce_sum(sqrt_energies , axis=1)
        sqrt_energy_a = sqrt_energy_sum[:,0]
        sqrt_energy_b = sqrt_energy_sum[:,1]
        
        print('sqrt_energy_sum',sqrt_energy_sum.shape)
        
        diff_sq = loss_per_vertex
        diff_sq_a = diff_sq[:,:,0]
        diff_sq_b = diff_sq[:,:,1]
        
        print('diff_sq_a',diff_sq_a.shape)
        
        e_for_loss = sqrt_energies
        esum_for_loss_a = sqrt_energy_a
        esum_for_loss_b = sqrt_energy_b
        
        if self.E_loss:
            e_for_loss = energies
            esum_for_loss_a = energy_a
            esum_for_loss_b = energy_b
            
        
        loss_a = tf.reduce_sum(diff_sq_a * e_for_loss[:,:,0],axis=1) / (esum_for_loss_a)
        loss_b = tf.reduce_sum(diff_sq_b * e_for_loss[:,:,1],axis=1) / (esum_for_loss_b)
        
        
        old_loss = (tf.reduce_sum(diff_sq_a * e_for_loss[:,:,0],axis=1) + tf.reduce_sum(diff_sq_b * e_for_loss[:,:,1],axis=1))/(esum_for_loss_a+esum_for_loss_b)
        
        print('loss_a',loss_a.shape)
        
        total_loss = (loss_a + loss_b)/2.
        
        response_a = tf.reduce_sum(prediction[:,:,0]*energy, axis=1) / energy_a
        response_b = tf.reduce_sum(prediction[:,:,1]*energy, axis=1) / energy_b
        
        
        print('response_a',response_a.shape)
        
        total_response = tf.concat([response_a , response_b], axis=0)
        
        self.mean_resolution, self.variance_resolution = self.normalise_response(total_response)
        
        self.total_loss = tf.reduce_mean(total_loss)
        
        sqrt_response_a = tf.reduce_sum(prediction[:,:,0]*sqrt_energies[:,:,0], axis=1) / sqrt_energy_a
        sqrt_response_b = tf.reduce_sum(prediction[:,:,1]*sqrt_energies[:,:,1], axis=1) / sqrt_energy_b
        
        sqrt_total_response = tf.concat([sqrt_response_a , sqrt_response_b], axis=0)
        
        self.mean_sqrt_resolution, self.variance_sqrt_resolution = self.normalise_response(sqrt_total_response)    
  
  
        if self.sum_loss:
            return self.total_loss
        return tf.reduce_mean(old_loss)


    def get_loss2(self):
        assert self._graph_output.shape[2] >= 2

        num_entries = tf.squeeze(self._placeholder_num_entries, axis=1)
        print('num_entries',num_entries.shape)
        energy = self._placeholder_other_features[:, :, 0]
        
        ###
        sqrt_energy = None
        if self.log_loss:
            sqrt_energy = tf.log(energy+1)
        else:
            sqrt_energy = tf.sqrt(energy)
        
        prediction = self._graph_output
        targets = self._placeholder_targets
        
        maxlen = self.max_entries
        #if self.use_seeds:
        #    energy=energy[:,0:-1] 
        #    targets = targets[:,0:-1,:]
        
        total_energy   = tf.reduce_sum(energy, axis=-1)
        
        print('total_energy',total_energy.shape)
        
        energies = targets * energy[:,:, tf.newaxis]
        energy_sums = tf.reduce_sum(energies , axis=1)
        energy_a = energy_sums[:,0]
        energy_b = energy_sums[:,1]
        
        print('energy_a',energy_a.shape)
        
        sqrt_energies = targets * sqrt_energy[:,:, tf.newaxis]
        
        print('sqrt_energies',sqrt_energies.shape)
        
        sqrt_energy_sum = tf.reduce_sum(sqrt_energies , axis=1)
        sqrt_energy_a = sqrt_energy_sum[:,0]
        sqrt_energy_b = sqrt_energy_sum[:,1]
        
        print('sqrt_energy_sum',sqrt_energy_sum.shape)
        
        diff_sq = (prediction[:,:,0:2] - targets) ** 2.
        diff_sq_a = diff_sq[:,:,0]
        diff_sq_b = diff_sq[:,:,1]
        
        print('diff_sq_a',diff_sq_a.shape)
        
        e_for_loss = sqrt_energies
        esum_for_loss_a = sqrt_energy_a
        esum_for_loss_b = sqrt_energy_b
        
        if self.E_loss:
            e_for_loss = energies
            esum_for_loss_a = energy_a
            esum_for_loss_b = energy_b
            
        
        loss_a = tf.reduce_sum(diff_sq_a * e_for_loss[:,:,0],axis=1) / (esum_for_loss_a)
        loss_b = tf.reduce_sum(diff_sq_b * e_for_loss[:,:,1],axis=1) / (esum_for_loss_b)
        
        
        old_loss = (tf.reduce_sum(diff_sq_a * e_for_loss[:,:,0],axis=1) + tf.reduce_sum(diff_sq_b * e_for_loss[:,:,1],axis=1))/(esum_for_loss_a+esum_for_loss_b)
        
        print('loss_a',loss_a.shape)
        
        total_loss = (loss_a + loss_b)/2.
        
        response_a = tf.reduce_sum(prediction[:,:,0]*energy, axis=1) / energy_a
        response_b = tf.reduce_sum(prediction[:,:,1]*energy, axis=1) / energy_b
        
        
        print('response_a',response_a.shape)
        
        total_response = tf.concat([response_a , response_b], axis=0)
        
        self.mean_resolution, self.variance_resolution = self.normalise_response(total_response)
        
        self.total_loss = tf.reduce_mean(total_loss)
        
        sqrt_response_a = tf.reduce_sum(prediction[:,:,0]*sqrt_energies[:,:,0], axis=1) / sqrt_energy_a
        sqrt_response_b = tf.reduce_sum(prediction[:,:,1]*sqrt_energies[:,:,1], axis=1) / sqrt_energy_b
        
        sqrt_total_response = tf.concat([sqrt_response_a , sqrt_response_b], axis=0)
        
        self.mean_sqrt_resolution, self.variance_sqrt_resolution = self.normalise_response(sqrt_total_response)    
  
  
        if self.sum_loss:
            return self.total_loss
        return tf.reduce_mean(old_loss) #+ tf.reduce_mean(0.1*tf.abs(1-self.mean_resolution)+0.1*self.variance_resolution)

    def _get_loss(self):
        if self.xentr_loss:
            return self.get_loss_xentropy()
        return self.get_loss2()
        

    def compute_output_seed_driven(self,_input,in_seed_idxs):
        
       
        net = _input

        seed_idxs = in_seed_idxs

        nfilters=24
        nspacefilters=30
        nspacedim=4
        
        
        feat = sparse_conv_collapse(net)
        
        for i in range(8):
            #feat = tf.Print(feat, [feat[0,2146,:]], 'space layer '+str(i),summarize=30)
            feat = sparse_conv_seeded3(feat, 
                       seed_idxs, 
                       nfilters=nfilters, 
                       nspacefilters=nspacefilters, 
                       nspacedim=nspacedim, 
                       seed_talk=True,
                       compress_before_propagate=True,
                       use_edge_properties=4)


        output = tf.layers.dense(feat, 3, activation=tf.nn.relu)
        
        return output
        
    def compute_output_full_adjecency(self,_input):
        
        pass
    
    
    
    def compute_output_seed_driven_neighbours(self,_input,seed_idxs):
        
        feat = sparse_conv_collapse(_input)
        
        
        feat_list=[]
        feat = sparse_conv_make_neighbors_simple(feat, 
                                      num_neighbors=16, 
                                      n_output=24,
                                      n_filters=[64,64,8], 
                                      edge_filters=[64,64,4],
                                      space_transformations=[32,4],
                                      train_global_space=False,)
        feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)
        feat_list.append(feat)
        
        
        filters = 6 * [24]
        aggregators = [2] + 3* [3] + 2 *[2]
        
        
        for i in range(len(filters)):
            feat = sparse_conv_aggregator_simple(feat, 
                       n_aggregators=aggregators[i], 
                       nfilters=filters[i], 
                       npropagate=filters[i],
                       nspacefilters=32, 
                       nspacedim=4,
                       use_edge_properties=4)
            feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)
            feat_list.append(feat)
        
        feat = tf.concat(feat_list, axis=-1)
        feat = tf.layers.dense(feat,32, activation=tf.nn.relu)
        feat = tf.layers.dense(feat,3, activation=tf.nn.relu)
        
        return feat
    
    def compute_output_dgcnn(self,_input,seeds,dropout=-1.):
        
        self.freeze_bn_after = None
        feat = sparse_conv_collapse(_input)
        feat = zero_out_by_energy(feat)
        
        feat = high_dim_dense(feat,16) #global transform to 3D
        feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)
        
        self.temp_feat_visualize = []
        feat = sparse_conv_edge_conv(feat,40,  [64,64,64])
        self.temp_feat_visualize.append(feat)
        feat_g = sparse_conv_global_exchange(feat)

        feat = tf.layers.dense(tf.concat([feat,feat_g],axis=-1),
                               64, activation=tf.nn.relu )

        feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)
        if dropout>0:
            feat = tf.layers.dropout(feat, rate=dropout,training=self.is_train)
        
        feat1 = sparse_conv_edge_conv(feat,40, [64,64,64])
        self.temp_feat_visualize.append(feat1)
        feat1_g = sparse_conv_global_exchange(feat1)
        feat1 = tf.layers.dense(tf.concat([feat1,feat1_g],axis=-1),
                                64, activation=tf.nn.relu )
        feat1 = tf.layers.batch_normalization(feat1,training=self.is_train, momentum=self.momentum)
        if dropout>0:
            feat1 = tf.layers.dropout(feat1, rate=dropout,training=self.is_train)
        
        feat2 = sparse_conv_edge_conv(feat1,40,[64,64,64])
        self.temp_feat_visualize.append(feat2)
        feat2_g = sparse_conv_global_exchange(feat2)
        feat2 = tf.layers.dense(tf.concat([feat2,feat2_g],axis=-1),
                                64, activation=tf.nn.relu )
        feat2 = tf.layers.batch_normalization(feat2,training=self.is_train,momentum=self.momentum)
        if dropout>0:
            feat2 = tf.layers.dropout(feat2, rate=dropout,training=self.is_train)
        
        feat3 = sparse_conv_edge_conv(feat2,40,[64,64,64])
        self.temp_feat_visualize.append(feat3)
        feat3 = tf.layers.batch_normalization(feat3,training=self.is_train, momentum=self.momentum)
        if dropout>0:
            feat3 = tf.layers.dropout(feat3, rate=dropout,training=self.is_train)
        
        #global_feat = tf.layers.dense(feat2,1024,activation=tf.nn.relu)
        #global_feat = max_pool_on_last_dimensions(global_feat, skip_first_features=0, n_output_vertices=1)
        #print('global_feat',global_feat.shape)
        #global_feat = tf.tile(global_feat,[1,feat.shape[1],1])
        #print('global_feat',global_feat.shape)
        
        feat = tf.concat([feat,feat1,feat2,feat_g,feat1_g,feat2_g,feat3],axis=-1)
        feat = tf.layers.dense(feat,32, activation=tf.nn.relu)
        feat = tf.layers.dense(feat,3, activation=tf.nn.relu)
        return feat
    
    def compute_output_neighbours(self,_input,seeds,dropout=0.01):
        
        self.freeze_bn_after = None
        feat = sparse_conv_collapse(_input)
        
        feat = tf.layers.batch_normalization(feat,training=self.is_train,momentum=self.momentum)
        
        feat_list=[]
        
        feat = sparse_conv_make_neighbors_simple(feat, 
                                      num_neighbors=16, 
                                      n_output=24,
                                      n_filters=[64,64,64,8], 
                                      edge_filters=[64,64,64,4],
                                      space_transformations=[32,4],
                                      train_global_space=False,)
        feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)
        if dropout>0:
            feat = tf.layers.dropout(feat, rate=dropout,training=self.is_train)
        feat = sparse_conv_global_exchange(feat)
        feat_list.append(feat)
        
        feat = sparse_conv_make_neighbors_simple(feat, 
                                      num_neighbors=24, 
                                      n_filters=[64,64,64,8], 
                                      edge_filters=[64,64,64,4],
                                      space_transformations=[32,4],
                                      train_global_space=False,)
        feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)
        if dropout>0:
            feat = tf.layers.dropout(feat, rate=dropout,training=self.is_train)
        feat = sparse_conv_global_exchange(feat)
        feat_list.append(feat)
        
        feat = sparse_conv_make_neighbors_simple(feat, 
                                      num_neighbors=24, 
                                      n_filters=[64,64,64,8], 
                                      edge_filters=[64,64,64,4],
                                      space_transformations=[32,4],
                                      train_global_space=False,)
        feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)
        if dropout>0:
            feat = tf.layers.dropout(feat, rate=dropout,training=self.is_train)
        feat = sparse_conv_global_exchange(feat)
        feat_list.append(feat)
        
        feat = tf.concat(feat_list, axis=-1)
        feat = tf.layers.dense(feat,64, activation=tf.nn.relu)
        feat = tf.layers.dense(feat,3, activation=tf.nn.relu)
        return feat
    
    
    def compute_output_make_neighbors_simple_multipass(self,_input,seeds,dropout=-0.01):
        feat = sparse_conv_collapse(_input)
        feat = tf.layers.batch_normalization(feat,training=self.is_train,momentum=self.momentum)
        feat_list=[]
        
        feat = sparse_conv_make_neighbors_simple_multipass(feat, 
                                      num_neighbors=16, 
                                      n_output=32,
                                      n_propagate=16,
                                      n_filters=   4*[20], 
                                      edge_filters=4*[2],
                                      space_transformations=[64,4],
                                      train_global_space=False,)
        feat = sparse_conv_global_exchange(feat)
        feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)
        feat = tf.layers.dense(feat,32, activation=tf.nn.tanh)
        feat_list.append(feat)
        feat = sparse_conv_make_neighbors_simple_multipass(feat, 
                                      num_neighbors=16, 
                                      n_output=32,
                                      n_propagate=16,
                                      n_filters=   4*[20], 
                                      edge_filters=4*[2],
                                      space_transformations=[32,4],
                                      train_global_space=False,)
        #feat = sparse_conv_global_exchange(feat)
        feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)
        feat_list.append(feat)
        
        
        feat = tf.concat(feat_list, axis=-1)
        feat = tf.layers.dense(feat,64, activation=tf.nn.relu)
        feat = tf.layers.dense(feat,3, activation=tf.nn.relu)
        return feat
        
        
    def compute_output_only_global_exchange(self,_input,seed_idxs):
        
        feat = sparse_conv_collapse(_input)
        feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum,center=False)
        global_feat = []
        depth = 13
        for i in range(depth):
            feat = sparse_conv_global_exchange(feat,
                                               expand_to_dims=-1,
                                               collapse_to_dims=42,
                                               learn_global_node_placement_dimensions=3)
            
            feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum,center=False)
            print('feat '+str(i), feat.shape)
            if i%2 or i==depth-1:
                global_feat.append(feat)
            
        feat = tf.concat(global_feat,axis=-1)
        print('feat concat', feat.shape)
        feat = tf.layers.dense(feat,32, activation=tf.nn.relu)
        feat = tf.layers.dense(feat,3, activation=tf.nn.relu)
        
        return feat
        
            
    def compute_output_moving_seeds3(self,_input,seeds,nspacedim,edge_multi=1):
        
        nfilters=32
        nspacefilters=64
        depth = 6
        
        
        feat = sparse_conv_collapse(_input)
        feat_list = []
        
        seeds=None
        for i in range(depth):
            n_seeds = 1
            if i>0:
                n_seeds=2
            if i>1:
                n_seeds=3

            feat,seeds = sparse_conv_moving_seeds3(feat, 
                             n_filters=nfilters, 
                             n_seeds=n_seeds, 
                             n_seed_dimensions=nspacedim,
                             seed_filters=[],
                             compress_before_propagate=True,
                             edge_multiplicity=edge_multi)
            
            #feat_list.append(feat)
        
        #feat =  tf.concat(feat_list,axis=-1)
        print('all feat',feat.shape)
        feat = tf.layers.dense(feat,42, activation=tf.nn.relu)
        feat = tf.layers.dense(feat,3, activation=tf.nn.relu)
        return feat
    
    def compute_output_moving_seeds4(self,_input,seeds,nspacedim,edge_multi=1):
        
        nfilters=42
        npropagate=8
        nspacefilters=128
        depth = 11
        
        
        feat = sparse_conv_collapse(_input)
        feat_list = []
        n_seeds = 4
        seeds=None
        for i in range(depth):

            feat,seeds = sparse_conv_moving_seeds4(feat, 
                             n_filters=nfilters, 
                             n_propagate=npropagate,
                             n_seeds=n_seeds, 
                             n_seed_dimensions=nspacedim,
                             seed_filters=[],
                             out_filters=[],
                             weight_filters=[],
                             edge_multiplicity=1)
            
            feat_list.append(feat)
        
        feat =  tf.concat(feat_list,axis=-1)
        print('all feat',feat.shape)
        feat = tf.layers.dense(feat,32, activation=tf.nn.relu)
        feat = tf.layers.dense(feat,3, activation=tf.nn.relu)
        return feat
    
    def compute_output_moving_seeds4_alt(self,_input,seeds,nspacedim,edge_multi=1):
        
        nfilters=32
        npropagate=4
        depth = 9
        
        
        feat = sparse_conv_collapse(_input)
        feat_list = []
        n_seeds=4
        seeds=None
        for i in range(depth):
            feat,seeds = sparse_conv_moving_seeds4(feat, 
                             n_filters=nfilters, 
                             n_propagate=npropagate,
                             n_seeds=n_seeds, 
                             n_seed_dimensions=nspacedim,
                             seed_filters=[],
                             compress_before_propagate=False,
                             edge_multiplicity=edge_multi)
            
            feat_list.append(feat)
        
        feat =  tf.concat(feat_list,axis=-1)
        print('all feat',feat.shape)
        feat = tf.layers.dense(feat,32, activation=tf.nn.relu)
        feat = tf.layers.dense(feat,3, activation=tf.nn.relu)
        return feat  
    
    def compute_output_moving_seeds4_generic(self,_input,seeds,nspacedim,edge_multi=1,
                                             depth=15,nfilters=32,n_seeds=4,npropagate=4,add_edge=False,
                                             weight_filters=[],seed_filters=[],out_filters=[],
                                             useNew=False,use_ms6=False):
        
        
        layer_func = sparse_conv_moving_seeds4
        
        if useNew:
            layer_func = sparse_conv_moving_seeds5
        if use_ms6:
            layer_func = sparse_conv_moving_seeds6
            
        feat = sparse_conv_collapse(_input)
        seeds=None
        feat_list=[]
        for i in range(depth):
            feat,seeds = layer_func(feat, 
                             n_filters=nfilters, 
                             n_propagate=npropagate,
                             n_seeds=n_seeds, 
                             n_seed_dimensions=nspacedim,
                             weight_filters=weight_filters,
                             seed_filters=seed_filters,
                             out_filters=out_filters,
                             edge_multiplicity=edge_multi,
                             add_egde_info=add_edge)
            
            feat_list.append(feat)
        
        feat =  tf.concat(feat_list,axis=-1)
        feat = tf.layers.dense(feat,32, activation=tf.nn.relu)
        feat = tf.layers.dense(feat,3, activation=tf.nn.relu)
        return feat   
    
    
    def compute_output_moving_seeds4_test(self,_input,seeds,nspacedim,edge_multi=1,
                                             depth=15,nfilters=32,n_seeds=4,npropagate=4,add_edge=False,
                                             weight_filters=[],seed_filters=[],out_filters=[],
                                             useNew=False):
        
        nspacedim=4
        edge_multi=1
        depth=10
        nfilters=32
        n_seeds=4
        npropagate=24
        
        layer_func = sparse_conv_moving_seeds5
            
        feat = sparse_conv_collapse(_input)
        feat = zero_out_by_energy(feat)
        seeds=None
        feat_list=[]
        for i in range(depth):
            feat,seeds = layer_func(feat, 
                             n_filters=nfilters, 
                             n_propagate=npropagate,
                             n_seeds=n_seeds, 
                             n_seed_dimensions=nspacedim,
                             #weight_filters=weight_filters,
                             #agg_feat_filters=[],#[32,16],
                             #agg_dim_filters=[],#[32,8],
                             #out_pre_feat_filters=[],#[32,16],
                             #out_feat_filters=[],#[32,16],
                             #out_agg_filters=[],#[32,32],
                             #out_filters=[32],
                             edge_multiplicity=edge_multi)
            
            feat_list.append(feat)
        
        feat =  tf.concat(feat_list,axis=-1)
        feat = tf.layers.dense(feat,32, activation=tf.nn.relu)
        feat = tf.layers.dense(feat,3, activation=tf.nn.relu)
        return feat   
    
    def compute_output_moving_seeds5(self,_input,seeds,
                                     nspacedim,
                                     edge_multi=1,
                                             depth=15,
                                             nfilters=32,
                                             n_seeds=4,
                                             npropagate=4):
        
        
        
        layer_func = sparse_conv_moving_seeds5
            
        feat = sparse_conv_collapse(_input)
        seeds=None
        feat_list=[]
        for i in range(depth):
            feat,seeds = layer_func(feat, 
                             n_filters=nfilters, 
                             n_propagate=npropagate,
                             n_seeds=n_seeds, 
                             n_seed_dimensions=nspacedim,
                             edge_multiplicity=edge_multi)
            
            feat_list.append(feat)
        
        feat =  tf.concat(feat_list,axis=-1)
        feat = tf.layers.dense(feat,32, activation=tf.nn.relu)
        feat = tf.layers.dense(feat,3, activation=tf.nn.relu)
        return feat   
    
    def compute_output_aggregator_simple(self,
                                         _input,seeds,dropout=-1,hipar=False,
                                         altconfig=False,weighted_agg_pos=False,
                                         propagate_half=False):
        
        feat = sparse_conv_collapse(_input)
        feat = zero_out_by_energy(feat)
        
        filters = 8 * [24]
        aggregators =[1] + [2] + 4* [3] + 2 *[2]
        
        if hipar:
            aggregators = [1] +  [6]  + 4 * [4]  + 2 *[3]
            filters =     [32] + [42] + 4 * [56] + 2 *[32]
        if altconfig:
            filters = 12 * [24]
            aggregators = 12*[1]
            
        propagate=filters
        if propagate_half:
            propagate = [f/2 for f in filters]
        
        feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)
        feat_list =[]
        use_do=dropout
        for i in range(len(filters)):
            if i == len(filters)-1:
                use_do=-1
            feat = sparse_conv_aggregator_simple(feat, 
                       n_aggregators=aggregators[i], 
                       nfilters=filters[i], 
                       npropagate=propagate[i],
                       nspacefilters=32, 
                       collapse_dropout=use_do,
                       expand_dropout=use_do, 
                       
                       return_aggregators=False,
                       input_aggregators=None,
                       weighted_aggregator_positions=weighted_agg_pos,
                       
                       is_training=self.is_train,
                       nspacedim=4,
                       use_edge_properties=4)
            feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)
            if dropout>0 and False:
                feat = tf.layers.dropout(feat, rate=dropout,training=self.is_train)
            feat_list.append(feat)
        
        feat=tf.concat(feat_list,axis=-1)
        if dropout>0:
            feat = tf.layers.dropout(feat, rate=dropout,training=self.is_train)
        feat = tf.layers.dense(feat,32, activation=tf.nn.relu)
        feat = tf.layers.dense(feat,3, activation=tf.nn.relu)
        return feat 
    
    def compute_output_aggregator_pass(self,
                                         _input,seeds,dropout=-1,hipar=False):
        
        feat = sparse_conv_collapse(_input)
        feat = zero_out_by_energy(feat)
        
        filters = 8 * [22]
        aggregators = 6 *[3] + 2 *[2]
        
        
        feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)
        feat_list =[]
        use_do=dropout
        agg = None
        for i in range(len(filters)):
            if i == len(filters)-1:
                use_do=-1
            feat, agg = sparse_conv_aggregator_simple(feat, 
                       n_aggregators=aggregators[i], 
                       nfilters=filters[i], 
                       npropagate=filters[i],
                       nspacefilters=24, 
                       collapse_dropout=use_do,
                       expand_dropout=use_do, 
                       
                       return_aggregators=True,
                       input_aggregators=agg,
                       
                       is_training=self.is_train,
                       nspacedim=4,
                       use_edge_properties=4)
            feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)
            agg = tf.layers.batch_normalization(agg,training=self.is_train, momentum=self.momentum)
            if dropout>0 and hipar:
                feat = tf.layers.dropout(feat, rate=dropout,training=self.is_train)
            feat_list.append(feat)
        
        feat=tf.concat(feat_list,axis=-1)
        if dropout>0:
            feat = tf.layers.dropout(feat, rate=dropout,training=self.is_train)
        feat = tf.layers.dense(feat,32, activation=tf.nn.relu)
        feat = tf.layers.dense(feat,3, activation=tf.nn.relu)
        return feat 
    
       
    def compute_output_moving_seeds_all_generic(self,_input,seeds,nspacedim,edge_multi,only_forward = False):
        
        nfilters=24
        nspacefilters=64
        depth = 5
        
        if edge_multi>1:
            nfilters=16
        
        feat = sparse_conv_collapse(_input)
        feat = zero_out_by_energy(feat)
        feat_list = []
        
        seeds=None
        for i in range(depth):
            n_seeds = 1
            multi=4
            if i>0:
                n_seeds=2
                multi=edge_multi
            if i>=2:
                n_seeds=4
            feat, seeds = sparse_conv_moving_seeds2(feat, 
                             n_filters=nfilters, 
                             n_spacefilters=nspacefilters, 
                             n_seed_dimensions=nspacedim, 
                             n_seeds=n_seeds, 
                             use_edge_properties=1,
                             compress_before_propagate=True,
                             seed_talk=True,
                             seed_positions=seeds,
                             edge_multiplicity=multi,
                             add_back_original = not only_forward)
            #feat = tf.Print(feat,[feat],'feat '+str(i), summarize=24)
            #feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum,
            #                                     center=False)
            #if i%2==0 or i == depth-1:
            feat_list.append(feat)
        
        feat =  tf.concat(feat_list,axis=-1)
        print('all feat',feat.shape)
        feat = tf.layers.dense(feat,3, activation=tf.nn.relu)
        return feat
    
    def compute_output_hidden_aggregators(self,_input,seeds,plusmean=False):
        feat = sparse_conv_collapse(_input)
        feat = zero_out_by_energy(feat)
        feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)
        
        aggregators = 11*[4]  
        filters =     11*[32] 
        propagate =   11*[20]
        pre_filters = 11*[[]]
        
        feat = sparse_conv_global_exchange(feat)
        feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)
        feat = high_dim_dense(feat,32, activation=tf.nn.tanh)
        feat_list=[]
        self.temp_feat_visualize = []
        for i in range(len(filters)):
            feat = sparse_conv_hidden_aggregators(feat,
                                                  aggregators[i],
                                                  n_filters=filters[i],
                                                  pre_filters=pre_filters[i],
                                                  n_propagate=propagate[i],
                                                  plus_mean=plusmean
                                                  )
            #self.temp_feat_visualize.append(xxx)
            feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)
            feat_list.append(feat)
            #feat = tf.layers.dropout(feat, rate=0.0005, training=self.is_train)
            
        feat =  tf.concat(feat_list,axis=-1)
        feat = tf.layers.dense(feat,48, activation=tf.nn.relu)
        feat = tf.layers.dense(feat,3, activation=tf.nn.relu)
        
        return feat
    
    def compute_output_hidden_aggregators_lowpara(self,_input,seeds,plusmean=False):
        feat = sparse_conv_collapse(_input)
        feat = zero_out_by_energy(feat)
        feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)
        
        aggregators = 11*[4]  
        filters =     11*[24] 
        propagate =   11*[10]
        pre_filters = 11*[[]]
        
        feat = sparse_conv_global_exchange(feat)
        feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)
        feat = high_dim_dense(feat,24, activation=tf.nn.tanh)
        feat_list=[]
        for i in range(len(filters)):
            feat = sparse_conv_hidden_aggregators(feat, 
                                                  aggregators[i],
                                                  n_filters=filters[i],
                                                  pre_filters=pre_filters[i],
                                                  n_propagate=propagate[i],
                                                  plus_mean=plusmean
                                                  )
            feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)
            feat_list.append(feat)
            #feat = tf.layers.dropout(feat, rate=0.0005, training=self.is_train)
            
        feat =  tf.concat(feat_list,axis=-1)
        feat = tf.layers.dense(feat,32, activation=tf.nn.relu)
        feat = tf.layers.dense(feat,3, activation=tf.nn.relu)
        
        return feat
    
        
    def compute_output_multi_neighbours(self,_input,seeds):
        feat = sparse_conv_collapse(_input)
        feat = zero_out_by_energy(feat)
        feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)
        
        filters=42
        propagate=18
        dimensions=4 
        neighbours=40
        
        feat_list = []
        for f in range(4):      
            feat = sparse_conv_global_exchange(feat)
            feat = high_dim_dense(feat,64, activation=tf.nn.tanh)
            feat = high_dim_dense(feat,64, activation=tf.nn.tanh)
            feat = high_dim_dense(feat,64, activation=tf.nn.tanh)
            feat = sparse_conv_multi_neighbours(feat,
                                       n_neighbours=neighbours,
                                       n_dimensions=dimensions,
                                       n_filters=filters,
                                       n_propagate=propagate,plus_mean=True)  
            feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)  
            feat_list.append(feat)
            #feat = tf.layers.dropout(feat, rate=0.0001, training=self.is_train)
        feat =  tf.concat(feat_list,axis=-1)
        print('all feat',feat.shape) 
        feat = tf.layers.dense(feat,128, activation=tf.nn.relu)
        feat = tf.layers.dense(feat,3, activation=tf.nn.relu)
        
        return feat
    
    def compute_output_single_neighbours(self,_input,seeds,plusmean=False):
        feat = sparse_conv_collapse(_input)
        feat = zero_out_by_energy(feat)
        feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)
        
        nfilters=48
        nprop=24
        if plusmean:
            nfilters=48
            nprop=22
        feat_list = []
        self.temp_feat_visualize=[]
        for f in range(4):      
            feat = sparse_conv_global_exchange(feat)  
            feat = high_dim_dense(feat,64, activation=tf.nn.tanh)
            feat = high_dim_dense(feat,64, activation=tf.nn.tanh)
            feat = high_dim_dense(feat,64, activation=tf.nn.tanh)
            #feat = tf.layers.dense(feat,56, activation=tf.nn.tanh,kernel_initializer=NoisyEyeInitializer) 
            feat = sparse_conv_multi_neighbours(feat,
                                       n_neighbours=40,
                                       n_dimensions=4,
                                       n_filters=nfilters,
                                       n_propagate=nprop,
                                       total_distance=True,
                                       plus_mean=plusmean)
            #self.temp_feat_visualize.append(xxx)
            feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)  
            feat_list.append(feat)
            #feat = tf.layers.dropout(feat, rate=0.0001, training=self.is_train)
        feat =  tf.concat(feat_list,axis=-1)
        print('all feat',feat.shape) 
        feat = tf.layers.dense(feat,128, activation=tf.nn.relu)
        feat = tf.layers.dense(feat,3, activation=tf.nn.relu)
        
        return feat
    
    def compute_output_single_neighbours_lowpara(self,_input,seeds,plusmean=False):
        feat = sparse_conv_collapse(_input)
        feat = zero_out_by_energy(feat)
        feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)
        
        nfilters=32
        nprop=16
        if plusmean:
            nfilters=32
            nprop=16
        feat_list = []
        for f in range(4):      
            feat = sparse_conv_global_exchange(feat)  
            feat = high_dim_dense(feat,42, activation=tf.nn.tanh)
            feat = high_dim_dense(feat,42, activation=tf.nn.tanh)
            feat = high_dim_dense(feat,42, activation=tf.nn.tanh)
            #feat = tf.layers.dense(feat,56, activation=tf.nn.tanh,kernel_initializer=NoisyEyeInitializer) 
            feat = sparse_conv_multi_neighbours(feat,
                                       n_neighbours=40,
                                       n_dimensions=4,
                                       n_filters=nfilters,
                                       n_propagate=nprop,
                                       total_distance=True,
                                       plus_mean=plusmean)  
            feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)  
            feat_list.append(feat)
            #feat = tf.layers.dropout(feat, rate=0.0001, training=self.is_train)
        feat =  tf.concat(feat_list,axis=-1)
        print('all feat',feat.shape) 
        feat = tf.layers.dense(feat,96, activation=tf.nn.relu)
        feat = tf.layers.dense(feat,3, activation=tf.nn.relu)
        
        return feat
     
    def compute_output_single_neighbours_conv(self,_input,seeds):
        feat = sparse_conv_collapse(_input)
        feat = zero_out_by_energy(feat)
        feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)
        
        feat_list = []
        feat = high_dim_dense(feat,64, activation=tf.nn.tanh)
        feat = high_dim_dense(feat,48, activation=tf.nn.tanh)  
        for f in range(4):      
            feat = sparse_conv_global_exchange(feat)
            feat = high_dim_dense(feat,48, activation=tf.nn.tanh)#,kernel_initializer=NoisyEyeInitializer)
            #feat = tf.layers.dense(feat,56, activation=tf.nn.tanh,kernel_initializer=NoisyEyeInitializer) 
            feat = sparse_conv_multi_neighbours(feat,
                                       n_neighbours=40,
                                       n_dimensions=4,
                                       n_filters=48,
                                       pre_filters=[],
                                       n_propagate=12,
                                       total_distance=True,
                                       individual_conv=True)  
            feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)  
            feat = high_dim_dense(feat,48, activation=tf.nn.tanh)#,kernel_initializer=NoisyEyeInitializer) 
            feat = high_dim_dense(feat,48, activation=tf.nn.tanh)#,kernel_initializer=NoisyEyeInitializer)
            feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)   
            feat_list.append(feat)
            #feat = tf.layers.dropout(feat, rate=0.0001, training=self.is_train)
        feat =  tf.concat(feat_list,axis=-1)
        print('all feat',feat.shape) 
        feat = tf.layers.dense(feat,128, activation=tf.nn.relu)
        feat = tf.layers.dense(feat,3, activation=tf.nn.relu)
        
        return feat    
    
    
    def compute_output_binning(self,_input,seeds):
        feat = sparse_conv_collapse(_input)
        feat = zero_out_by_energy(feat)
        feat = tf.layers.batch_normalization(feat, training=self.is_train)
        feat, indices = construct_binning16(feat)
        print(feat.shape)
        
        feat_list=[]
        
        feat = sparse_conv_global_exchange_binned(feat)
        feat = tf.layers.conv3d(feat, 32, [1, 1, 1], activation=tf.nn.tanh, padding='same')
        feat = tf.layers.batch_normalization(feat, training=self.is_train)
        
        feat_list.append(feat)
        
        feat = tf.layers.conv3d(feat, 28, [6, 6, 1], activation=tf.nn.tanh, padding='same')
        feat = tf.layers.conv3d(feat, 28, [1, 1, 3], activation=tf.nn.tanh, padding='same')
        feat = tf.layers.batch_normalization(feat, training=self.is_train)
        
        feat_list.append(feat)
        
        feat = sparse_conv_global_exchange_binned(feat)
        feat = tf.layers.conv3d(feat, 32, [1, 1, 1], activation=tf.nn.tanh, padding='same')
        feat = tf.layers.batch_normalization(feat, training=self.is_train)
        
        
        feat = tf.layers.conv3d(feat, 28, [6, 6, 1], activation=tf.nn.tanh, padding='same')
        feat = tf.layers.conv3d(feat, 28, [1, 1, 3], activation=tf.nn.tanh, padding='same')
        feat = tf.layers.batch_normalization(feat, training=self.is_train)
        
        feat_list.append(feat)
        
        feat = tf.layers.conv3d(feat, 28, [6, 6, 1], activation=tf.nn.tanh, padding='same')
        feat = tf.layers.conv3d(feat, 28, [1, 1, 3], activation=tf.nn.tanh, padding='same')
        feat = tf.layers.batch_normalization(feat, training=self.is_train)
        
        feat_list.append(feat)
        
        #feat = tf.concat(feat_list,axis=-1)
        vert_list=[]
        for feat in feat_list:
            pass
            assert int(feat.shape[-1])>=4
            print('feat a',feat.shape)
            feat = tf.reshape(feat, [int(feat.shape[0]), 16, 16, 20, 4, int(int(feat.shape[-1])/4)])
            feat = tf.gather_nd(feat, indices)
            vert_list.append(feat)
            print('feat b',feat.shape)
            
        feat = tf.concat(vert_list, axis=-1)
        print('feat all',feat.shape)
        feat = tf.layers.dense(feat,128, activation=tf.nn.relu)
        return feat
    
    def compute_output_binning20(self,_input,seeds):
        feat = sparse_conv_collapse(_input)
        feat = zero_out_by_energy(feat)
        feat = tf.layers.batch_normalization(feat, training=self.is_train)
        feat, indices = construct_binning20(feat)
        print(feat.shape)
        
        feat_list=[]
        
        feat = sparse_conv_global_exchange_binned(feat)
        feat = tf.layers.conv3d(feat, 20, [1, 1, 1], activation=tf.nn.tanh, padding='same')
        feat = tf.layers.batch_normalization(feat, training=self.is_train)
        
        feat_list.append(feat)
        
        feat = tf.layers.conv3d(feat, 24, [8, 8, 1], activation=tf.nn.tanh, padding='same')
        feat = tf.layers.conv3d(feat, 20, [1, 1, 3], activation=tf.nn.tanh, padding='same')
        feat = tf.layers.batch_normalization(feat, training=self.is_train)
        
        feat_list.append(feat)
        
        feat = sparse_conv_global_exchange_binned(feat)
        feat = tf.layers.conv3d(feat, 20, [1, 1, 1], activation=tf.nn.tanh, padding='same')
        feat = tf.layers.batch_normalization(feat, training=self.is_train)
        
        
        feat = tf.layers.conv3d(feat, 24, [8, 8, 1], activation=tf.nn.tanh, padding='same')
        feat = tf.layers.conv3d(feat, 20, [1, 1, 3], activation=tf.nn.tanh, padding='same')
        feat = tf.layers.batch_normalization(feat, training=self.is_train)
        
        feat_list.append(feat)
        
        feat = tf.layers.conv3d(feat, 24, [8, 8, 1], activation=tf.nn.tanh, padding='same')
        feat = tf.layers.conv3d(feat, 20, [1, 1, 3], activation=tf.nn.tanh, padding='same')
        feat = tf.layers.batch_normalization(feat, training=self.is_train)
        
        feat_list.append(feat)
        
        #feat = tf.concat(feat_list,axis=-1)
        vert_list=[]
        for feat in feat_list:
            pass
            assert int(feat.shape[-1])>=1
            print('feat a',feat.shape)
            feat = tf.reshape(feat, [int(feat.shape[0]), 20, 20, 20, 1, int(int(feat.shape[-1])/1)])
            feat = tf.gather_nd(feat, indices)
            vert_list.append(feat)
            print('feat b',feat.shape)
            
        feat = tf.concat(vert_list, axis=-1)
        print('feat all',feat.shape)
        feat = tf.layers.dense(feat,128, activation=tf.nn.relu)
        return feat
    
    def compute_output_binning20_2(self,_input,seeds):
        feat = sparse_conv_collapse(_input)
        feat = zero_out_by_energy(feat)
        feat = tf.layers.batch_normalization(feat, training=self.is_train)
        feat, indices = construct_binning20(feat)
        print(feat.shape)
        
        feat_list=[]
        
        feat = sparse_conv_global_exchange_binned(feat)
        feat = tf.layers.conv3d(feat, 20, [1, 1, 1], activation=tf.nn.tanh, padding='same')
        feat_in = tf.layers.batch_normalization(feat, training=self.is_train)
        
        feat = feat_in
        
        for i in range(8):
            
            feat = tf.layers.conv3d(feat, 14, [7, 7, 1], activation=tf.nn.tanh, padding='same')
            feat = tf.layers.conv3d(feat, 14, [1, 1, 3], activation=tf.nn.tanh, padding='same')
            feat = tf.layers.batch_normalization(feat, training=self.is_train)
            
            feat_list.append(feat)
        
        
        #feat = tf.concat(feat_list,axis=-1)
        vert_list=[]
        for feat in feat_list:
            pass
            assert int(feat.shape[-1])>=1
            print('feat a',feat.shape)
            feat = tf.reshape(feat, [int(feat.shape[0]), 20, 20, 20, 1, int(int(feat.shape[-1])/1)])
            feat = tf.gather_nd(feat, indices)
            vert_list.append(feat)
            print('feat b',feat.shape)
            
        feat = tf.concat(vert_list, axis=-1)
        print('feat all',feat.shape)
        feat = tf.layers.dense(feat,128, activation=tf.nn.relu)
        return feat
              

    def _compute_output(self):
        
        
        feat = self._placeholder_other_features
        print("feat",feat.shape)
        space_feat = self._placeholder_space_features
        local_space_feat = self._placeholder_space_features_local
        num_entries = self._placeholder_num_entries
        n_batch = space_feat.shape[0]
        
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
            
        nrandom = 1
        random_seeds = tf.random_uniform(shape=(int(n_batch),nrandom),minval=0,maxval=2102,dtype=tf.int64)
        print('random_seeds',random_seeds.shape)
        seeds = tf.concat([self._placeholder_seed_indices,random_seeds],axis=-1)
        seeds = tf.transpose(seeds,[1,0])
        seeds = tf.random_shuffle(seeds)
        seeds = tf.transpose(seeds,[1,0])
        seeds = self._placeholder_seed_indices
        print('seeds',seeds.shape)
        
        net_in = construct_sparse_io_dict(feat, space_feat, local_space_feat,
                                          tf.squeeze(num_entries))
        
        net = sparse_conv_normalise(net_in,log_energy=True)
        
        #net = sparse_conv_add_simple_seed_labels(net,seeds)
        
        #simple_input = tf.concat([space_feat,local_space_feat,feat],axis=-1)
        #output=self.compute_output_seed_driven(net,seeds)#self._placeholder_seed_indices)
        # output = self.compute_output_seed_driven_neighbours(net,seeds)
        #output = self.compute_output_neighbours(net,self._placeholder_seed_indices)
        
        #PAPER
        if self.get_variable_scope() == 'dgcnn':
            output = self.compute_output_dgcnn(net,self._placeholder_seed_indices)
            
        if self.get_variable_scope() == 'dgcnn_linE':
            self.E_loss=True
            output = self.compute_output_dgcnn(net,self._placeholder_seed_indices)
            
        
        ### PAPER MODEL  
        elif self.get_variable_scope() == 'hidden_aggregators_plusmean'  :
            self.sum_loss=True
            output = self.compute_output_hidden_aggregators(net,seeds,plusmean=True) 
            
            
        elif self.get_variable_scope() == 'hidden_aggregators_plusmean_linE'  :
            self.sum_loss=True
            self.E_loss=True
            output = self.compute_output_hidden_aggregators(net,seeds,plusmean=True) 
            
            
            
             
             
        ### PAPER MODEL
        elif self.get_variable_scope() == 'single_neighbours_plusmean':
            output = self.compute_output_single_neighbours(net,seeds,plusmean=True)  
        elif self.get_variable_scope() == 'single_neighbours_plusmean_v100':
            output = self.compute_output_single_neighbours(net,seeds,plusmean=True)  
            
            
        elif self.get_variable_scope() == 'single_neighbours_plusmean_linE':
            self.E_loss=True
            output = self.compute_output_single_neighbours(net,seeds,plusmean=True)  
        elif self.get_variable_scope() == 'single_neighbours_plusmean_v100_linE':
            self.E_loss=True
            output = self.compute_output_single_neighbours(net,seeds,plusmean=True)  
            
            
            
        ##PAPER MODEL 
        elif self.get_variable_scope() == 'output_binning20_2':
            output = self.compute_output_binning20_2(net,seeds) 
            
            
        ##PAPER MODEL 
        elif self.get_variable_scope() == 'output_binning20_2_xEntr':
            self.xentr_loss=True
            output = self.compute_output_binning20_2(net,seeds) 
            
            
        elif self.get_variable_scope() == 'output_binning20_2_linE':
            self.E_loss=True
            output = self.compute_output_binning20_2(net,seeds) 
        
           
            
            
        #elif self.get_variable_scope() == 'only_global_exchange':
        #    output = self.compute_output_only_global_exchange(net,self._placeholder_seed_indices)
        #elif self.get_variable_scope() == 'seed_driven':
        #    output = self.compute_output_seed_driven(net, self._placeholder_seed_indices)
        else:
            output = tf.layers.dense(sparse_conv_collapse(net),3)
            
        #output=self.compute_output_seed_driven(net,self._placeholder_seed_indices)
        #output=self.compute_output_full_adjecency(_input)
        output = tf.layers.dense(output,2)
        output = tf.nn.softmax(output)
        
        self._graph_temp = tf.reduce_sum(output[:,:,:], axis=1)/2679.

        return output

    def get_variable_scope(self):
        return self.config_name


    def _construct_graphs(self):
        
        self.learningrate_scheduler.create_exponential_wiggle(self.start_learning_rate, 
                                                              self.start_learning_rate/80, 
                                                              end_exp_iterations=2000000* 150/self.batch_size,
                                                              scaler=25000* 150/self.batch_size,
                                                              wiggle_frequency=0.1,
                                                              n_points=50) 
        
        
        
        with tf.variable_scope(self.get_variable_scope()):
            self.initialized = True
            self.weight_init_width=1e-6

            self.make_placeholders()

            self._graph_output = self._compute_output()

            # self._graph_temp = tf.nn.softmax(self.__graph_logits)

            self._graph_loss = self._get_loss()
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            print(update_ops)
            with tf.control_dependencies(update_ops):
                self._graph_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self._graph_loss)

            # Repeating, maybe there is a better way?
            self._graph_summary_loss = tf.summary.scalar('loss', self._graph_loss)
            self._graph_summaries = tf.summary.merge([self._graph_summary_loss, 
                                                      tf.summary.scalar('mean-res', self.mean_resolution), 
                                                      tf.summary.scalar('variance-res', self.variance_resolution),
                                                      tf.summary.scalar('mean-res-sqrt', self.mean_sqrt_resolution), 
                                                      tf.summary.scalar('variance-res-sqrt', self.variance_sqrt_resolution), 
                                                      tf.summary.scalar('learning-rate', self.learning_rate)])

            self._graph_summary_loss_validation = tf.summary.scalar('Validation Loss', self._graph_loss)
            self._graph_summaries_validation = tf.summary.merge([self._graph_summary_loss_validation])

    def get_losses(self):
        print("Hello, world!")
        return self._graph_loss
