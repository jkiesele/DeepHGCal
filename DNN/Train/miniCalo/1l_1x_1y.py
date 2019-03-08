

from DeepJetCore.training.training_base import training_base
import keras
from keras.layers import Dense, Conv1D, ZeroPadding3D,Conv3D,Dropout, Flatten, Convolution2D,Conv3D, merge, Convolution1D, Conv2D, LSTM, LocallyConnected2D
from keras.layers.pooling import MaxPooling3D
from keras.models import Model
from keras.layers.core import Reshape, Masking, Permute, RepeatVector
from keras.layers.pooling import MaxPooling2D, MaxPooling3D,AveragePooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate,Add,Multiply
from keras.layers.noise import GaussianDropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Cropping3D
from keras.regularizers import l2
import keras.backend as K

import tensorflow as tf

from Layers import Multiply_feature,simple_correction_layer,Sum3DFeatureOne,Sum3DFeaturePerLayer, SelectEnergyOnly, ReshapeBatch, ScalarMultiply, Log_plus_one, Clip, Print, Reduce_sum, Sum_difference_squared
############# just a placeholder

from keras.layers import LeakyReLU
leaky_relu_alpha=0.001

from tools import miniCalo_preprocess,miniCalo_global_correction,load_global_correction




def resnet_like_3D(Inputs,nclasses,nregressions,dropoutRate=0.05,momentum=0.6, nodemulti=32, l2_lambda=0.0001, use_dumb_bias=False):
    
    x = miniCalo_preprocess(Inputs[0])
    
    x = Flatten()(x)
    print(x.shape)
    
    id = Dense(1, trainable=False)(x)
    

    #x = Dense(1,name="pre_E",use_bias=False,kernel_initializer='ones')(x)
    
    
    predictE = miniCalo_global_correction(x)

    predictions = [id,predictE]
    
    model = Model(inputs=Inputs, outputs=predictions)
    return model




#also dows all the parsing
train=training_base(testrun=False,resumeSilently=False,renewtokens=True)



from Losses import calo_loss_rel, huber_loss_calo, acc_calo_relative_rms,acc_calo_relative_rms_10,acc_calo_relative_rms_20,acc_calo_relative_rms_70

usemetrics=['mean_squared_error',acc_calo_relative_rms,
            acc_calo_relative_rms_10,
            acc_calo_relative_rms_20,
            acc_calo_relative_rms_70,]
usefinetuneloss=huber_loss_calo #huber_loss_calo #low_value_msq #'mean_squared_error' # huber_loss_calo #huber_loss_relative #'mean_squared_error' #huber_loss_relative #huber_loss_relative #low_value_msq


additionalplots=['val_E_acc_calo_relative_rms',
                 'val_E_acc_calo_relative_rms_10',
                 'val_E_acc_calo_relative_rms_20',
                 'val_E_acc_calo_relative_rms_70']

if not train.modelSet():
    
    train.setModel(resnet_like_3D)
    
    if True:
        train.keras_model = \
    load_global_correction(
        '/afs/cern.ch/user/j/jkiesele/work/DeepLearning/FCChh/DeepHGCal/DNN/modules/global_correction_layers.h5',
        train.keras_model)
    
    train.compileModel(learningrate=0.01,#will be overwritten anyway
                       clipnorm=1,
                   loss=['mean_absolute_error',acc_calo_relative_rms],metrics=usemetrics,
                   loss_weights=[1e-7, 1.]) #ID, en
    

#print(train.keras_model.summary())
#exit()




batchsize=5000
clipnorm=1

#train.trainedepoches=0
#K.set_value(train.keras_model.optimizer.lr, lrs.lr)

from training_scheduler import Learning_sched, scheduled_training

verbose=1
learn=[]

learn.append(Learning_sched(lr=1e-4,     nepochs=1,    batchsize=1000,  funccall=None, loss_weights=[1e-7, 1.],  en_loss='mean_squared_error'))
learn.append(Learning_sched(lr=3e-4,     nepochs=3,    batchsize=5000,  funccall=None, loss_weights=[1e-7, 1.],  en_loss=calo_loss_rel))
learn.append(Learning_sched(lr=1e-4,     nepochs=10,   batchsize=20000,  funccall=None, loss_weights=[1e-7, 1.],  en_loss=calo_loss_rel))
learn.append(Learning_sched(lr=1e-5,     nepochs=10,   batchsize=50000,  funccall=None, loss_weights=[1e-7, 1.],  en_loss=calo_loss_rel))
learn.append(Learning_sched(lr=1e-6,     nepochs=85,   batchsize=50000,  funccall=None, loss_weights=[1e-7, 1.],  en_loss=calo_loss_rel))



scheduled_training( learn, train,
                    clipnorm=1,usemetrics=usemetrics, 
                    
                    checkperiod=5,
                    verbose=verbose,
                    additional_plots=additionalplots)












