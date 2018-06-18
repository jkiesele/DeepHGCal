import tensorflow as tf
from ops.sparse_conv import *
from ops.neighbors import *


def printAndEval(prtext, x, sess):
    
    print(prtext,x.shape)
    result = sess.run(x)
    print(result)

#BxEx3
test_spatial_global = tf.constant([[[1, 2, 3],
                                    [1.1, 2.1, 3.1],
                                    [1, 1.9, 3],
                                    [0, 3, 2],
                                    [0, 4, 2]],
                                    [[11, 12, 13],
                                     [11.1, 12.1, 13.1],
                                     [11, 11.9, 13],
                                     [10, 13, 12],
                                     [10, 14, 12]]],dtype=tf.float32)

test_spatial_local = tf.constant([[[10, 30],
                                   [20, 50],
                                   [10, 60],
                                   [20, 10],
                                   [00, 20]],
                                   [[110, 130],
                                    [120, 150],
                                    [110, 160],
                                    [120, 110],
                                    [100, 120]]],dtype=tf.float32)

#BxExF
test_features = tf.constant([[[.1, .2],
                              [.2, .3],
                              [.1, .4],
                              [.2, .2],
                              [.0, .4]],
                              [[1.1, 1.2],
                               [1.2, 1.3],
                               [1.1, 1.4],
                               [1.2, 1.2],
                               [1.0, 1.4]]],dtype=tf.float32)


distancetozero  =  tf.constant([[[.1],
                              [.2],
                              [.1],
                              [.2],
                              [.0]],
                              [[1.1],
                               [1.2],
                               [1.1],
                               [1.2],
                               [1.0]]],dtype=tf.float32)


n_batch=2
n_max_entries=5
output_all=8
#assumes euclidean
print('input shape spat features: ',test_spatial_global.shape)

with tf.Session() as sess:
    
    matrix = nearest_neighbor_matrix(test_spatial_global,3)
    result = sess.run(matrix)
    print(result)
    
    
    indices = sort_last_dim_tensor(distancetozero)
    printAndEval('indices',indices,sess)
    
    gathered_feat=tf.gather_nd(test_features, indices)
    printAndEval('gathered_feat',gathered_feat,sess)
    
   
