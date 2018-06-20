from keras import backend as K

global_loss_list = {}

def huber(x):
    clip_delta=10.
    import tensorflow as tf
    
    cond  = tf.abs(x) < clip_delta
    
    squared_loss = tf.square(x)
    linear_loss  = 2* clip_delta * (tf.abs(x) - 0.5 * clip_delta)
    
    return tf.where(cond, squared_loss, linear_loss)

def huber_loss(y_true, y_pred):
    return huber(y_true-y_pred)
    

global_loss_list['huber_loss']=huber_loss

def huber_loss_calo(y_true, y_pred):
    '''
    calo-like huber loss with quadratic until around 10% relative difference for inputs around 100
    '''
    import tensorflow as tf
    scaleddiff=(y_true-y_pred)/tf.sqrt(y_true+0.1)
    return huber(scaleddiff)
    

global_loss_list['huber_loss_calo']=huber_loss_calo


def acc_calo_relative_rms(y_true, y_pred):
    import tensorflow as tf
    return tf.sqrt(K.mean( tf.square((y_true-y_pred)/( y_true+0.1)) , axis=-1)) * 100
    
global_loss_list['acc_calo_relative_rms']=acc_calo_relative_rms

def loss_logcoshClipped(y_true, y_pred):
    """
    scaled log cosh with nan detection
    scale set to 100, after which it becomes linearish
    """
    
    scaler=30
    #from tensorflow import where, greater, abs, zeros_like, exp
    import tensorflow as tf
    
    def cosh(x):
        return (tf.exp(x) + tf.exp(-x)) / 2
    
    def scaledlogcoshdiff(x,y,scale):
        return tf.log(cosh(x/scale - y/scale))
                     
    def cleaned_scaledlogcoshdiff(x,y,scale):
        logcosh=scaledlogcoshdiff(x,y,scale)
        return tf.where(tf.is_nan(logcosh), tf.zeros_like(logcosh)+scaler*scaler*tf.abs(x-y), logcosh)
    
    
    return K.mean(2*scaler*scaler*cleaned_scaledlogcoshdiff(y_true,y_pred,scaler) , axis=-1)
    
global_loss_list['loss_logcoshClipped']=loss_logcoshClipped


def loss_Calo_logcoshClipped(y_true, y_pred):
    
    scaler=.1 #get linear after 10% resolution

    import tensorflow as tf
    
    def cosh(x):
        calccosh=(tf.exp(x) + tf.exp(-x)) / 2
        return tf.where(tf.is_nan(calccosh), tf.square(x) , calccosh)
    
    def scaledrellogcoshdiff(x,y,scale):
        out = tf.log(cosh( (x - y)/(scale* tf.sqrt(tf.abs(x)+1) ) ))
        out = tf.where(tf.is_nan(out), tf.square(x - y) , out)
        return tf.where(tf.is_inf(out), tf.square(x - y) , out)

    return K.mean(scaledrellogcoshdiff(y_true,y_pred,scaler) , axis=-1)
    

global_loss_list['loss_Calo_logcoshClipped']=loss_Calo_logcoshClipped



def mean_squared_mixed_logarithmic_error(y_true, y_pred):
    from tensorflow import where, greater
    
    scaler = 0.05
    
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.mean(K.square(first_log - second_log)+
                  scaler* K.square(y_pred - y_true), axis=-1)

global_loss_list['mean_squared_mixed_logarithmic_error'] = mean_squared_mixed_logarithmic_error

def mean_squared_logarithmic_error(y_true, y_pred):
    from tensorflow import where, greater
    
    scaler=0.1
    
    first_log = K.log(K.clip(scaler*y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(scaler*y_true, K.epsilon(), None) + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)

global_loss_list['mean_squared_logarithmic_error'] = mean_squared_logarithmic_error

def loss_NLL(y_true, x):
    """
    This loss is the negative log likelyhood for gaussian pdf.
    See e.g. http://bayesiandeeplearning.org/papers/BDL_29.pdf for details
    Generally it might be better to even use Mixture density networks (i.e. more complex pdf than single gauss, see:
    https://publications.aston.ac.uk/373/1/NCRG_94_004.pdf
    """
    x_pred = x[:, 1:]
    x_sig = x[:, :1]
    return K.mean(0.5 * K.log(K.square(x_sig)) + K.square(x_pred - y_true) / K.square(x_sig) / 2., axis=-1)


global_loss_list['loss_NLL'] = loss_NLL


def loss_NLL_mod(y_true, x):
    """
    Modified loss_NLL, such that very large deviations have reduced impact
    """
    import numpy
    from tensorflow import where, greater, abs, zeros_like, exp

    x_pred = x[:, 1:]
    x_sig = x[:, :1]

    gaussianCentre = K.square(x_pred - y_true) / K.square(x_sig) / 2
    outerpart = K.log(abs(x_pred - y_true) / abs(x_sig) - exp(-1.))

    res = where(greater(gaussianCentre, 4), 0.5 * K.log(abs(x_sig)) + outerpart,
                0.5 * K.log(K.square(x_sig)) + gaussianCentre)

    # remove non truth-matched
    # res=where(greater(y_true,0),res,zeros_like(y_true))


    # penalty for too small sigmas
    # res=where(greater(x_sig,1),res,res+10000*K.square(x_sig-1) )

    return K.mean(res, axis=-1)


global_loss_list['loss_NLL_mod'] = loss_NLL_mod


def loss_modRelMeanSquaredError(y_true, x_pred):
    """
    testing - name is also wrong depending on commit..
    """

    res = K.square((x_pred - y_true)) / (y_true + 3.)
    # res=where(greater(y_true,0.0001),res,zeros_like(y_true))

    return K.mean(res, axis=-1)


global_loss_list['loss_modRelMeanSquaredError'] = loss_modRelMeanSquaredError


def loss_relAndAbsMeanSquaredError(y_true, x_pred):
    """
    testing - name is also wrong depending on commit..
    """

    from tensorflow import where, greater, abs, zeros_like, exp



    res = K.square(x_pred - y_true) + 10*K.square((x_pred - y_true) / (y_true + .03))
    # res=where(greater(y_true,0.0001),res,zeros_like(y_true))

    return K.mean(res, axis=-1)


global_loss_list['loss_relAndAbsMeanSquaredError'] = loss_relAndAbsMeanSquaredError

def loss_relMeanSquaredErrorScaled(y_true, x):
    """
    testing - name is also wrong depending on commit..
    """
    
    from tensorflow import where, greater, abs, zeros_like, exp
    
    x_sig = x[:,:1]
    x_pred = x[:,1:]
    scaledtruth=y_true/500 - 1
    
    res=(0.00001* K.square(x_sig - x_pred) + K.square((x_pred - scaledtruth) ) / K.square(scaledtruth+ 1.01))
    #res=where(greater(y_true,0.0001),res,zeros_like(y_true))
    
    return K.mean(res ,    axis=-1)


global_loss_list['loss_relMeanSquaredErrorScaled']=loss_relMeanSquaredErrorScaled


def loss_meanSquaredError(y_true, x):
    """
    testing - name is also wrong depending on commit..
    """
    
    from tensorflow import where, greater, abs, zeros_like, exp
    
    x_pred = x[:,1:]
    x_sig = x[:,:1]
    
    
    res=0.0000001* K.square(x_sig - x_pred - y_true) + K.square((x_pred - y_true) ) 
    #res=where(greater(y_true,0.0001),res,zeros_like(y_true))
    
    return K.mean(res ,    axis=-1)


global_loss_list['loss_meanSquaredError']=loss_meanSquaredError



def loss_logcoshScaled(y_true, x):
    """
    testing - name is also wrong depending on commit..
    """
    
    from tensorflow import where, greater, abs, zeros_like, exp
    
    x_pred = x[:,1:]
    x_sig = x[:,:1]
    def cosh(x):
        return (K.exp(x) + K.exp(-x)) / 2
    
    return K.mean(0.001*K.square(x_sig))   + K.mean(K.log(cosh(x_pred - y_true/500+1)), axis=-1)
    


global_loss_list['loss_logcoshScaled']=loss_logcoshScaled


def loss_meanSquaredErrorScaled(y_true, x):
    """
    testing - name is also wrong depending on commit..
    """
    
    from tensorflow import where, greater, abs, zeros_like, exp
    
    x_pred = x[:,1:]
    x_sig = x[:,:1]
    scaledtruth=y_true/500 -1
    
    res=0.0000001* K.square(x_sig) + K.square((x_pred - scaledtruth) ) 
    #res=where(greater(y_true,0.0001),res,zeros_like(y_true))
    
    return K.mean(res ,    axis=-1)


global_loss_list['loss_meanSquaredErrorScaled']=loss_meanSquaredErrorScaled


def accuracy_None(y_true, x):
    return 0


def accuracy_SigPred(y_true, x):
    """
    testing - name is also wrong depending on commit..
    """

    from tensorflow import where, greater, abs, zeros_like, exp

    x_pred = x[:, 1:]
    x_sig = x[:, :1]

    return K.mean(x_sig / y_true, axis=-1)


# The below is to use multiple gaussians for regression

## https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation/blob/master/MDN-DNN-Regression.ipynb
## these next three functions are from Axel Brando and open source, but credits need be to https://creativecommons.org/licenses/by-sa/4.0/ in case we use it

def log_sum_exp(x, axis=None):
    """Log-sum-exp trick implementation"""
    x_max = K.max(x, axis=axis, keepdims=True)
    return K.log(K.sum(K.exp(x - x_max),
                       axis=axis, keepdims=True)) + x_max


def mean_log_Gaussian_like(y_true, parameters):
    """Mean Log Gaussian Likelihood distribution
    Note: The 'c' variable is obtained as global variable
    """

    # Note: The output size will be (c + 2) * m = 6
    c = 1  # The number of outputs we want to predict
    m = 2  # The number of distributions we want to use in the mixture
    components = K.reshape(parameters, [-1, c + 2, m])
    mu = components[:, :c, :]
    sigma = components[:, c, :]
    alpha = components[:, c + 1, :]
    alpha = K.softmax(K.clip(alpha, 1e-8, 1.))

    exponent = K.log(alpha) - .5 * float(c) * K.log(2 * np.pi) \
               - float(c) * K.log(sigma) \
               - K.sum((K.expand_dims(y_true, 2) - mu) ** 2, axis=1) / (2 * (sigma) ** 2)

    log_gauss = log_sum_exp(exponent, axis=1)
    res = - K.mean(log_gauss)
    return res


def mean_log_LaPlace_like(y_true, parameters):
    """Mean Log Laplace Likelihood distribution
    Note: The 'c' variable is obtained as global variable
    """
    # Note: The output size will be (c + 2) * m = 6
    c = 1  # The number of outputs we want to predict
    m = 2  # The number of distributions we want to use in the mixture
    components = K.reshape(parameters, [-1, c + 2, m])
    mu = components[:, :c, :]
    sigma = components[:, c, :]
    alpha = components[:, c + 1, :]
    alpha = K.softmax(K.clip(alpha, 1e-2, 1.))

    exponent = K.log(alpha) - float(c) * K.log(2 * sigma) \
               - K.sum(K.abs(K.expand_dims(y_true, 2) - mu), axis=1) / (sigma)

    log_gauss = log_sum_exp(exponent, axis=1)
    res = - K.mean(log_gauss)
    return res
