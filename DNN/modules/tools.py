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

from Layers import Sum3DFeatureOne,Sum3DFeaturePerLayer, Create_per_layer_energies,SelectEnergyOnly, ReshapeBatch, Multiply, Log_plus_one, Clip, SelectFeatureOnly, Print, Reduce_sum, Multiply_feature
from tensorflow.contrib.learn.python.learn import trainable

def create_full_calo_image(Inputs, dropoutRate, momentum, trainable=True):
    
    ecalhits=Inputs[0]
    hcalhits=Inputs[1]

    #scale down layer info
    hcalhits=Multiply_feature(0, 0.001)(hcalhits)
    ecalhits=Multiply_feature(0, 0.001)(ecalhits)
    
    
    leaky_relu_alpha=0.001
    #ecalhits = SelectEnergyOnly()(ecalhits)
    
    ecalhits = Multiply(1./1000.)(ecalhits)
    #17x17x10x2
    #hcalhits = SelectEnergyOnly()(hcalhits)
    hcalhits = Multiply(1./1000.)(hcalhits)
    
    
    hcalhits_energy = SelectEnergyOnly()(hcalhits)
    ecalhits_energy = SelectEnergyOnly()(ecalhits)
    
    
    ecalhits       = Conv3D(4,kernel_size=(1,1,1),strides=(1,1,1), 
                      kernel_initializer='lecun_uniform',
                      #use_bias=False,
                      trainable=trainable,
                      padding='same',name='pre_ecalhits1')(ecalhits)  
    ecalhits = LeakyReLU(alpha=leaky_relu_alpha)(ecalhits)
    
    ecalhits_energy = Conv3D(1,kernel_size=(2,2,1),strides=(2,2,1), 
                      kernel_initializer='ones',
                      use_bias=False,
                      trainable=False,
                      padding='same',name='dumb_ecal_energy')(ecalhits_energy) 
                      
    
    ecalhits = Conv3D(7,kernel_size=(2,2,1),strides=(2,2,1), 
                      kernel_initializer=keras.initializers.random_normal(1./2., 1./16.),
                      #use_bias=False,
                      trainable=trainable,
                      padding='same',name='pre_ecalhits2')(ecalhits)  
    ecalhits = LeakyReLU(alpha=leaky_relu_alpha)(ecalhits) 
    ecalhits = Dropout(dropoutRate)(ecalhits)            
             
    hcalhits  = Conv3D(7,kernel_size=(1,1,1),strides=(1,1,1), 
                      kernel_initializer=keras.initializers.random_normal(1./8., 1./32.),
                      #use_bias=False,
                      trainable=trainable,
                      padding='same',name='pre_hcalhits1')(hcalhits)
    hcalhits = LeakyReLU(alpha=leaky_relu_alpha)(hcalhits)   
    hcalhits = Dropout(dropoutRate)(hcalhits)  
    
    hcalhits =  Concatenate()([hcalhits_energy,hcalhits])
    ecalhits =  Concatenate()([ecalhits_energy,ecalhits])
                        
     
    fullcaloimage = Concatenate(axis=-2,name='fullcaloimage')([ecalhits,hcalhits])
    if momentum>0:
        fullcaloimage = BatchNormalization(momentum=momentum,name="norm_fullimage",epsilon=0.1)(fullcaloimage)
                                       
    #fullcaloimage = LeakyReLU(alpha=leaky_relu_alpha)(fullcaloimage) 
    
    
    return  fullcaloimage


def create_per_layer_energies(Inputs):
    return Concatenate(name='ecal_hcal_concat')([Create_per_layer_energies()(Inputs[0]),Create_per_layer_energies()(Inputs[1])])
    

def sum_layer_energies(perlayerenergies, strides, rows):
    reordered=[perlayerenergies[strides*i : strides*(i+1)] for i in range(rows)]
    outlist=[]
    for r in reordered:
        outlist.append(Add()(r))
    return outlist
        
    
    

def create_conv_resnet(x, name,
                       kernel_dumb, strides_dumb, 
                       nodes_lin,
                       nodes_nonlin, kernel_nonlin_a, kernel_nonlin_b, lambda_reg,
                       leaky_relu_alpha=0.001,
                       dropout=0.000001,
                       normalize_dumb=False,
                       nodes_dumb=1,dumb_trainable=False,
                       lin_trainable=False,
                       use_dumb_bias=True):
    
    
    
    x_dumb = SelectFeatureOnly(0)(x)
    
    
    x_dumb = Conv3D(nodes_dumb,kernel_size=kernel_dumb,strides=strides_dumb, name=name+'_dumb',
                      kernel_initializer='ones',
                      use_bias=use_dumb_bias,
                      trainable=dumb_trainable,
                      padding='valid')(x_dumb)
    x_dumb = LeakyReLU(alpha=leaky_relu_alpha)(x_dumb)
      
    x_lin = Conv3D(nodes_lin,kernel_size=kernel_dumb,strides=strides_dumb, name=name+'_lin',
                      kernel_initializer=keras.initializers.random_normal(0.0, 1e-3),
                      trainable=lin_trainable,
                      #kernel_regularizer=l2(0.1*lambda_reg),
                      padding='valid')(x)
    x_lin = LeakyReLU(alpha=leaky_relu_alpha)(x_lin)
    x_lin = Dropout(0.1*dropout)(x_lin)     
     
    #non-linear
    x_resnet = Dropout(2*dropout)(x)   
    x_resnet = Conv3D(nodes_nonlin,kernel_size=kernel_nonlin_a,strides=(1,1,1), name=name+'_res_a',
                      kernel_initializer=keras.initializers.random_normal(0.0, 1e-3),
                      activation='relu',
                      trainable=False,
                      kernel_regularizer=l2(lambda_reg),
                      padding='same')(x_resnet)
    #x_resnet = BatchNormalization(momentum=0.6,name=name+'_res_a_bn',trainable=False)(x_resnet) 
    x_resnet = Dropout(2*dropout)(x_resnet)    
                 
    x_resnet = Conv3D(nodes_nonlin,kernel_size=kernel_nonlin_b,strides=(1,1,1), name=name+'_res_b',
                      kernel_initializer=keras.initializers.random_normal(0.0, 1e-3),
                      activation='relu',
                      trainable=False,
                      kernel_regularizer=l2(lambda_reg),
                      padding='same')(x_resnet)
    #x_resnet = BatchNormalization(momentum=0.6,name=name+'_res_b_bn',trainable=False)(x_resnet) 
    x_resnet = Dropout(2*dropout)(x_resnet)    
                      
    x_resnet = Conv3D(nodes_lin,kernel_size=kernel_dumb,strides=strides_dumb, name=name+'_res_c',
                      kernel_initializer=keras.initializers.random_normal(0.0, 1e-3),
                      activation='relu',
                      trainable=False,
                      kernel_regularizer=l2(lambda_reg),
                      padding='valid')(x_resnet)
    
    x_lin = Add()([x_lin,x_resnet])
    x_dumb = Concatenate()([x_dumb,x_lin])
    
    return x_dumb




