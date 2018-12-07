import numpy as np
import tensorflow as tf

def euclidean_squared(A, B):
    """
    Returns euclidean distance between two batches of shape [B,N,F] and [B,M,F] where B is batch size, N is number of
    examples in the batch of first set, M is number of examples in the batch of second set, F is number of spatial
    features.

    Returns:
    A matrix of size [B, N, M] where each element [i,j] denotes euclidean distance between ith entry in first set and
    jth in second set.

    """

    shape_A = A.get_shape().as_list()
    shape_B = B.get_shape().as_list()

    assert (A.dtype == tf.float32 or A.dtype == tf.float64) and (B.dtype == tf.float32 or B.dtype == tf.float64)
    assert len(shape_A) == 3 and len(shape_B) == 3
    assert shape_A[0] == shape_B[0]  # and shape_A[1] == shape_B[1]

    # just exploit broadcasting
    B_trans = tf.expand_dims(B, axis=1)
    A_exp = tf.expand_dims(A, axis=2)
    print(B_trans.shape, A_exp.shape)
    diff = A_exp - B_trans
    distance = tf.reduce_sum(tf.square(diff), axis=-1)
    # to avoid rounding problems and keep it strict positive
    distance = tf.abs(distance)
    return distance


def euclidean_squared_2(A, B):
    """
    Returns euclidean distance between two batches of shape [B,N,F] and [B,M,F] where B is batch size, N is number of
    examples in the batch of first set, M is number of examples in the batch of second set, F is number of spatial
    features.

    Returns:
    A matrix of size [B, N, M] where each element [i,j] denotes euclidean distance between ith entry in first set and
    jth in second set.

    """

    shape_A = A.get_shape().as_list()
    shape_B = B.get_shape().as_list()

    assert (A.dtype == tf.float32 or A.dtype == tf.float64) and (B.dtype == tf.float32 or B.dtype == tf.float64)
    assert len(shape_A) == 3 and len(shape_B) == 3
    assert shape_A[0] == shape_B[0]  # and shape_A[1] == shape_B[1]

    # Finds euclidean distance using property (a-b)^2 = a^2 + b^2 - 2ab
    sub_factor = -2 * tf.matmul(A, tf.transpose(B, perm=[0, 2, 1]))  # -2ab term
    dotA = tf.expand_dims(tf.reduce_sum(A * A, axis=2), axis=2)  # a^2 term
    dotB = tf.expand_dims(tf.reduce_sum(B * B, axis=2), axis=1)  # b^2 term
    return sub_factor + dotA + dotB

A = np.array(
    [
        [
            [-81,1],
            [1,98]
        ]
    ]
)


first = tf.constant(A, dtype=tf.float64)
first = euclidean_squared_2(first, first)
second = tf.constant(A, dtype=tf.float64)
second = euclidean_squared(second,second)

with tf.Session() as sess:
    a, b = sess.run([first, second])

    print(a)

    print(b)