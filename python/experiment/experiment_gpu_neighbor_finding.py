import tensorflow as tf
import numpy as np





def get_euclidean_two_datasets(A, B):
    sub_factor = -2 * tf.matmul(space_features, tf.transpose(space_features, perm=[0, 2, 1]))
    dotA = tf.expand_dims(tf.reduce_sum(space_features * space_features, axis=2), axis=2)
    dotB = tf.expand_dims(tf.reduce_sum(space_features * space_features, axis=2), axis=1)

    print(sub_factor.get_shape().as_list())
    print(dotA.get_shape().as_list())
    print(dotB.get_shape().as_list())

    return sub_factor + dotA + dotB


def find_nearest_neighbor_matrix(D, k):
    _, temp = tf.nn.top_k(-D, k)
    print(temp.get_shape().as_list())
    return temp


batch_size = 500

space_features = tf.placeholder(dtype=tf.float32,shape=[batch_size, 2000, 3])
D = get_euclidean_two_datasets(space_features, space_features)
N = find_nearest_neighbor_matrix(D, k=10)

with tf.Session() as sess:
    A = np.random.randn(batch_size, 2000, 3)
    print("Running...")
    v = sess.run([N], feed_dict={space_features: A})
    print("Done!")
    print(v)
