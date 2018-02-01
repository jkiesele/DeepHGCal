from keras import backend as K

global_loss_list = {}





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


def loss_relMeanSquaredError(y_true, x_pred):
    """
    testing - name is also wrong depending on commit..
    """

    from tensorflow import where, greater, abs, zeros_like, exp



    res = K.square((x_pred - y_true) / (y_true + .01))
    # res=where(greater(y_true,0.0001),res,zeros_like(y_true))

    return K.mean(res, axis=-1)


global_loss_list['loss_relMeanSquaredError'] = loss_relMeanSquaredError


def loss_relAndAbsMeanSquaredError(y_true, x_pred):
    """
    testing - name is also wrong depending on commit..
    """

    from tensorflow import where, greater, abs, zeros_like, exp



    res = K.square(x_pred - y_true) + K.square((x_pred - y_true) / (y_true + .01))
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
