

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

from Layers import simple_correction_layer,Sum3DFeatureOne,Sum3DFeaturePerLayer, SelectEnergyOnly, ReshapeBatch, ScalarMultiply, Log_plus_one, Clip, Print, Reduce_sum, Sum_difference_squared
############# just a placeholder

from keras.layers import LeakyReLU
leaky_relu_alpha=0.001

from tools import create_full_calo_image, create_per_layer_energies, create_conv_resnet

def freeze_all_batchnorms(model):
    for layer in model.layers:
       if isinstance(layer, keras.layers.normalization.BatchNormalization):
           layer._per_input_updates = {}
           layer.trainable = False
           print('frozen batch norm ',layer.name)
           
           
def simple_global_correction(x):
    x = simple_correction_layer()(x)



def resnet_like_3D(Inputs,nclasses,nregressions,dropoutRate=0.05,momentum=0.6, nodemulti=32, l2_lambda=0.0001, use_dumb_bias=False):
    
    x = Flatten()(Inputs[0])
    print(x.shape)
    
    id = Dense(1, trainable=False)(x)
    

    x = Dense(1,name="E",use_bias=False)(x)
    x = simple_correction_layer(name="last_correction")(x)

    predictions = [id,x]
    
    model = Model(inputs=Inputs, outputs=predictions)
    return model




#also dows all the parsing
train=training_base(testrun=False,resumeSilently=False,renewtokens=True)
from DeepJetCore.DataCollection import DataCollection
orig=DataCollection.generator
def _adapted_generator(self):
    for t in orig(self):
        x=t[0]
        y=t[1]
        y[0]*=0
        yield (x,y)
#DataCollection.generator=_adapted_generator


additionalplots=[]

from Losses import huber_loss_calo, acc_calo_relative_rms, huber_loss_relative, huber_loss,low_value_msq,acc_calo_relative_rms_50,acc_calo_relative_rms_100,acc_calo_relative_rms_500,acc_calo_relative_rms_1000
usemetrics=['mean_squared_error']
usefinetuneloss=huber_loss_calo #huber_loss_calo #low_value_msq #'mean_squared_error' # huber_loss_calo #huber_loss_relative #'mean_squared_error' #huber_loss_relative #huber_loss_relative #low_value_msq


if not train.modelSet():
    
    train.setModel(resnet_like_3D)
    
    train.compileModel(learningrate=0.01,#will be overwritten anyway
                       clipnorm=1,
                   loss=['mean_absolute_error','mean_squared_error'],metrics=usemetrics,
                   loss_weights=[1e-7, 1.]) #ID, en
    

#print(train.keras_model.summary())
#exit()


model,history = train.trainModel(nepochs=100, 
                                 batchsize=5000,
                                 checkperiod=1,
                                 verbose=1,
                                 additional_plots=additionalplots)
        
exit()






