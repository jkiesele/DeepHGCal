import numpy as np
import tensorflow as tf
from ops.neighbors import sort_last_dim_tensor
import time


shape = [100, 700, 100]
num_entries = tf.placeholder(dtype=tf.int64, shape=[shape[0]])
input = tf.placeholder(dtype=tf.float32, shape=shape)

op = tf.expand_dims(tf.cast(tf.sequence_mask(lengths=num_entries, maxlen=shape[1]), tf.float32), axis=2) * input
op = sort_last_dim_tensor(input)


with tf.Session() as sess:
    while True:
        B = np.random.uniform(100, shape[1], size=shape[0])
        A = np.random.uniform(size=shape)
        print("Hello, world!")

        start = time.time()
        sess.run(op, feed_dict={input : A, num_entries : B})
        end = time.time()

        print(end-start)
