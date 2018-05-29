import tensorflow as tf
import numpy as np





X=tf.placeholder(dtype=tf.float32, shape=[1, 3])
Y=tf.placeholder(dtype=tf.float32, shape=[1, 3])
N = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=X)

with tf.Session() as sess:
    x = np.array([[0,0,1]])
    y = np.array([[1e5,1e5,1e3]])
    print("Running...")
    v = sess.run([N], feed_dict={X: x, Y: y})
    print("Done!")
    print(v)
