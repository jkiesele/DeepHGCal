import tensorflow as tf



def gauss(x):
    return tf.exp(-1* x*x)

def gauss_of_lin(x):
    return tf.exp(-1*(tf.abs(x)))

def gauss_times_linear(x):
    return tf.exp(-6.*tf.sqrt(tf.abs(x)+1e-4))*x*3.*36 

def asymm_falling(x):
    return tf.exp(-2.*x*x)*x*2.

def sinc(x):
    return tf.sin(x)/x

def open_tanh(x):
    return 0.9*tf.nn.tanh(x)+0.1*x


# more sophisticated ones

def multi_dim_edge_activation(x):
    return tf.concat([gauss_of_lin(x[...,0:1]), gauss_times_linear(x[...,1:])], axis=-1)