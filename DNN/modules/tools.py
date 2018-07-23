import keras
from keras.layers import Dense, Conv1D, ZeroPadding3D,Conv3D,Dropout, Flatten, Convolution2D,Conv3D, merge, Convolution1D, Conv2D, LSTM, LocallyConnected2D
from keras.layers.pooling import MaxPooling3D
from keras.models import Model
from keras.layers.core import Reshape, Masking, Permute, RepeatVector
from keras.layers.pooling import MaxPooling2D, MaxPooling3D,AveragePooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate,Add
from keras.layers.noise import GaussianDropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Cropping3D
from keras.regularizers import l2
import keras.backend as K

from Layers import Sum3DFeatureOne,Sum3DFeaturePerLayer, SelectEnergyOnly, ReshapeBatch, Multiply, Log_plus_one, Clip, SelectFeatureOnly, Print, Multiply_feature
from tensorflow.contrib.learn.python.learn import trainable

def create_full_calo_image(Inputs, dropoutRate, momentum, trainable=True):
    
    ecalhits=Inputs[0]
    hcalhits=Inputs[1]

    #scale down layer info
    hcalhits=Multiply_feature(0,0.001)(hcalhits)
    ecalhits=Multiply_feature(0,0.001)(ecalhits)
    
    leaky_relu_alpha=0.001
    #ecalhits = SelectEnergyOnly()(ecalhits)
    
    ecalhits = Multiply(0.1)(ecalhits)
    #17x17x10x2
    #hcalhits = SelectEnergyOnly()(hcalhits)
    hcalhits = Multiply(0.1)(hcalhits)
    
    
    initialisewith='lecun_uniform'
    if not trainable:
        initialisewith=keras.initializers.random_normal(0.0, 1e-6)
    
    ecalhits       = Conv3D(8,kernel_size=(1,1,1),strides=(1,1,1), 
                      kernel_initializer='lecun_uniform',
                      #use_bias=False,
                      trainable=trainable,
                      padding='same',name='pre_ecalhits1')(ecalhits)  
    ecalhits = LeakyReLU(alpha=leaky_relu_alpha)(ecalhits)
    
    ecalhits = Dropout(dropoutRate)(ecalhits)         
    ecalcompressed = Conv3D(8,kernel_size=(2,2,1),strides=(2,2,1), 
                      kernel_initializer=initialisewith,
                      #use_bias=False,
                      trainable=trainable,
                      padding='same',name='pre_ecalhits5')(ecalhits)  
    ecalcompressed = LeakyReLU(alpha=leaky_relu_alpha)(ecalcompressed)          
             
    hcalhits  = Conv3D(8,kernel_size=(1,1,1),strides=(1,1,1), 
                      kernel_initializer=initialisewith,
                      #use_bias=False,
                      trainable=trainable,
                      padding='same',name='pre_hcalhits5')(hcalhits)
    hcalhits = LeakyReLU(alpha=leaky_relu_alpha)(hcalhits)     
                        
     
    fullcaloimage = Concatenate(axis=-2,name='fullcaloimage')([ecalcompressed,hcalhits])
    if momentum>0:
        fullcaloimage = BatchNormalization(momentum=momentum,name="norm_fullimage",epsilon=0.1)(fullcaloimage)
                                       
    #fullcaloimage = LeakyReLU(alpha=leaky_relu_alpha)(fullcaloimage) 
    fullcaloimage = Dropout(dropoutRate)(fullcaloimage)
    
    return  fullcaloimage


def create_per_layer_energies(Inputs):
    
    perlayerenergies=[]
    for i in range(8):
        perlayerenergies.append(Sum3DFeaturePerLayer(i)(Inputs[0]))
    for i in range(10):
        perlayerenergies.append(Sum3DFeaturePerLayer(i)(Inputs[1]))
    
    allayerenergies=Concatenate(name='perlayeren_concat')(perlayerenergies)
    allayerenergies = Multiply(1/500.)(allayerenergies)  
    return allayerenergies

def create_conv_resnet(x, name,
                       nodes_dumb, kernel_dumb, strides_dumb,
                       nodes_lin,
                       nodes_nonlin, kernel_nonlin_a, kernel_nonlin_b, lambda_reg,
                       leaky_relu_alpha=0.001,
                       dropout=0.000001,
                       normalize_dumb=False):
    
    
    selecteddumb=[]
    for i in range(min(nodes_dumb, x.shape[4])):
        selecteddumb.append(SelectFeatureOnly(i)(x))
    x_dumb = Concatenate()(selecteddumb)

    x_dumb = Conv3D(nodes_dumb,kernel_size=kernel_dumb,strides=strides_dumb, name=name+'_dumb',
                      kernel_initializer=keras.initializers.random_normal(0.0, 0.5*20/nodes_dumb),
                      use_bias=False,
                      padding='valid')(x_dumb)
    x_dumb = LeakyReLU(alpha=leaky_relu_alpha)(x_dumb)
    if normalize_dumb:
        x_dumb = BatchNormalization(momentum=0.6,name=name+"_dumb_norm",epsilon=0.1)(x_dumb) 
      
    x_lin = Conv3D(nodes_lin,kernel_size=kernel_dumb,strides=strides_dumb, name=name+'_lin',
                      kernel_initializer=keras.initializers.random_normal(0.0, 1e-6),
                      trainable=False,
                      kernel_regularizer=l2(0.1*lambda_reg),
                      padding='valid')(x)
    x_lin = LeakyReLU(alpha=leaky_relu_alpha)(x_lin)
    x_lin = Dropout(dropout)(x_lin)     
     
    #non-linear
    x_resnet = Conv3D(nodes_nonlin,kernel_size=kernel_nonlin_a,strides=(1,1,1), name=name+'_res_a',
                      kernel_initializer='lecun_uniform',
                      activation='relu',
                      trainable=False,
                      kernel_regularizer=l2(lambda_reg),
                      padding='same')(x)
    x_resnet = BatchNormalization(momentum=0.6)(x_resnet) 
    x_resnet = Dropout(dropout)(x_resnet)    
                 
    x_resnet = Conv3D(nodes_nonlin,kernel_size=kernel_nonlin_b,strides=(1,1,1), name=name+'_res_b',
                      kernel_initializer='lecun_uniform',
                      activation='relu',
                      trainable=False,
                      kernel_regularizer=l2(lambda_reg),
                      padding='same')(x_resnet)
    x_resnet = BatchNormalization(momentum=0.6)(x_resnet) 
    x_resnet = Dropout(dropout)(x_resnet)    
                      
    x_resnet = Conv3D(nodes_lin+nodes_dumb,kernel_size=kernel_dumb,strides=strides_dumb, name=name+'_res_c',
                      kernel_initializer=keras.initializers.random_normal(0.0, 1e-6),
                      activation='relu',
                      trainable=False,
                      kernel_regularizer=l2(lambda_reg),
                      padding='valid')(x_resnet)
    x_resnet = Dropout(dropout)(x_resnet)     
    
    x_out = Concatenate()([x_dumb,x_lin]) 
    x_out = Add()([x_out,x_resnet])
     
    
    return x_out




