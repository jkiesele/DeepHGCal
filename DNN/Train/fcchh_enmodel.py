

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

from Layers import Sum3DFeatureOne,Sum3DFeaturePerLayer, SelectEnergyOnly, ReshapeBatch, Multiply, Log_plus_one, Clip, Print
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

def ultra_simple_model_3D(Inputs,nclasses,nregressions,dropoutRate=0.05,momentum=0.6, nodemulti=32, l2_lambda=0.0001):
    
    fullcaloimage = create_full_calo_image(Inputs, dropoutRate=0.0001, momentum=momentum)
    
    x = fullcaloimage

    x = Conv3D(1*nodemulti,kernel_size=(2, 2, 1),strides=(2,2,1), name='conv1',
                      kernel_initializer=keras.initializers.random_normal(0, 0.05),
                      bias_initializer=keras.initializers.random_normal(0.01, 0.01),
                      kernel_regularizer=l2(l2_lambda),
                      border_mode='valid')(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)                      
    x = Dropout(dropoutRate)(x)  
    x = Conv3D(2*nodemulti,kernel_size=(4, 4, 1),strides=(4,4,1), name='conv2',
                      kernel_initializer=keras.initializers.random_normal(0, 0.05),
                      bias_initializer=keras.initializers.random_normal(0.01, 0.01),
                      kernel_regularizer=l2(l2_lambda),
                      border_mode='valid')(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)   
    x = Dropout(dropoutRate)(x) 
    x = Conv3D(2*nodemulti,kernel_size=(1, 1, 4),strides=(1,1,4), name='conv3',
                      kernel_initializer=keras.initializers.random_normal(0, 0.05),
                      bias_initializer=keras.initializers.random_normal(0.01, 0.01),
                      kernel_regularizer=l2(l2_lambda),
                      border_mode='valid')(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)   
    x = Dropout(dropoutRate)(x) 
    x = Conv3D(2*nodemulti,kernel_size=(2, 2, 2),strides=(2,2,2), name='conv6',
                      kernel_initializer=keras.initializers.random_normal(0, 0.05),
                      bias_initializer=keras.initializers.random_normal(0.01, 0.01),
                      kernel_regularizer=l2(l2_lambda),
                      border_mode='valid')(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)   
    x = BatchNormalization(momentum=momentum,name="norm_after_conv_block",epsilon=0.1,
                                       center=True,scale=True)(x)
    x = Dropout(dropoutRate)(x) 
    
    
    ## now a more id like part
    idx = fullcaloimage
    idx = Conv3D(nodemulti,kernel_size=(4, 4, 1),strides=(1,1,1), name='conv2a',
                      kernel_initializer=keras.initializers.random_normal(0, 0.005),
                      kernel_regularizer=l2(10*l2_lambda),
                      border_mode='valid')(idx)
    idx = LeakyReLU(alpha=leaky_relu_alpha)(idx)   
    idx = Dropout(0.05)(idx) 
    idx = Conv3D(nodemulti,kernel_size=(1, 1, 4),strides=(1,1,1), name='conv3a',
                      kernel_initializer=keras.initializers.random_normal(0, 0.005),
                      kernel_regularizer=l2(10*l2_lambda),
                      border_mode='valid')(idx)
    idx = LeakyReLU(alpha=leaky_relu_alpha)(idx)   
    idx = Dropout(0.05)(idx) 
    idx = Conv3D(nodemulti,kernel_size=(2, 2, 2),strides=(2,2,2), name='conv6a',
                      kernel_initializer=keras.initializers.random_normal(0, 0.005),
                      kernel_regularizer=l2(10*l2_lambda),
                      border_mode='valid')(idx)
    idx = LeakyReLU(alpha=leaky_relu_alpha)(idx)                     
    idx = Dropout(0.05)(idx)
    idx = Conv3D(nodemulti/2,kernel_size=(2, 2, 2),strides=(2,2,2), name='conv7a',
                      kernel_initializer=keras.initializers.random_normal(0, 0.005),
                      kernel_regularizer=l2(10*l2_lambda),
                      border_mode='valid')(idx)
    idx = LeakyReLU(alpha=leaky_relu_alpha)(idx)                     
    idx = BatchNormalization(momentum=momentum,name="norm_after_conv_blocka",epsilon=0.1,
                                       center=True,scale=True)(idx)
    idx = Dropout(dropoutRate)(idx)
    
    idx=Flatten()(idx)
    x = Flatten()(x)
    x = Concatenate()([x,idx])
    
    #x = Dropout(dropoutRate)(x)
    
    x = Dense(32, kernel_initializer='lecun_uniform',name='dense3',activation='relu')(x)
    x = Dropout(dropoutRate)(x) 
    x = Dense(32, kernel_initializer='lecun_uniform',name='dense4',activation='relu')(x)
    
    predictE=Dense(1, activation='linear',kernel_initializer='lecun_uniform',name='pre_pred_E')(x)
    predictE = Clip(0.001,2.1)(predictE)
    predictE=Multiply(500,name='pred_E')(predictE)
    predictID=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    predictions = [predictID,predictE]
    
    model = Model(inputs=Inputs, outputs=predictions)
    return model
    

def resnet_like_3D(Inputs,nclasses,nregressions,dropoutRate=0.05,momentum=0.6, nodemulti=32, l2_lambda=0.0001):
    
    fullcaloimage = create_full_calo_image(Inputs, dropoutRate=0.001, momentum=momentum,trainable=True)
    allayers = create_per_layer_energies(Inputs)
    
    x = fullcaloimage

    allayerenergies= RepeatVector(17*17)(allayers)
    allayerenergies = Reshape((17,17,18,1))(allayerenergies)
    allayerenergies = Multiply(1./(17.*17.))(allayerenergies)
    
    x = Concatenate()([allayerenergies,x])
    #make the really dumb part that gets corrections
    
    
    x = create_conv_resnet(x, name='conv_block_1',
                       nodes_dumb=16, kernel_dumb=(2,2,1), strides_dumb=(2,2,1),
                       nodes_lin=4,
                       nodes_nonlin=32, kernel_nonlin_a=(3,3,2), kernel_nonlin_b=(2,2,3), lambda_reg=l2_lambda,
                       leaky_relu_alpha=0.01,
                       dropout=dropoutRate)
    x = Dropout(0.1*dropoutRate)(x)
    
    x = create_conv_resnet(x, name='conv_block_2',
                       nodes_dumb=16, kernel_dumb=(4,4,1), strides_dumb=(4,4,1),
                       nodes_lin=4,
                       nodes_nonlin=32, kernel_nonlin_a=(4,4,2), kernel_nonlin_b=(2,2,3), lambda_reg=l2_lambda,
                       leaky_relu_alpha=0.01, 
                       dropout=dropoutRate)
    x = Dropout(0.1*dropoutRate)(x)
    
    x = create_conv_resnet(x, name='conv_block_3',
                       nodes_dumb=16, kernel_dumb=(1,1,4), strides_dumb=(1,1,4),
                       nodes_lin=4,
                       nodes_nonlin=32, kernel_nonlin_a=(3,3,2), kernel_nonlin_b=(2,2,4), lambda_reg=l2_lambda,
                       leaky_relu_alpha=0.01,
                       dropout=dropoutRate)
    x = Dropout(0.1*dropoutRate)(x)
    
    x = create_conv_resnet(x, name='conv_block_4',
                       nodes_dumb=16, kernel_dumb=(2,2,2), strides_dumb=(2,2,2),
                       nodes_lin=4,
                       nodes_nonlin=24, kernel_nonlin_a=(3,3,3), kernel_nonlin_b=(3,3,3), lambda_reg=l2_lambda,
                       leaky_relu_alpha=0.01,
                       dropout=dropoutRate,
                       normalize_dumb=True)
    
    
    x = Dropout(0.1*dropoutRate)(x)
    x = Flatten()(x)
    
    # x = Concatenate()([allayers,x])
    
    x_lin = Dense(15, kernel_initializer=keras.initializers.random_normal(0.0, 1e-6),name='dense1_lin', trainable=False)(x)
    x_lin = LeakyReLU(alpha=leaky_relu_alpha)(x_lin)
    x_dumb = Dense(24, kernel_initializer='lecun_uniform',name='dense1_dumb', use_bias=False)(x)
    x_dumb = LeakyReLU(alpha=0.001)(x_dumb)
    x_dumb = Dense(16, kernel_initializer='lecun_uniform',name='dense2_dumb', use_bias=False)(x_dumb)
    x_dumb = LeakyReLU(alpha=0.001)(x_dumb)
    x = Concatenate()([x_lin,x_dumb])
    
    predictE=Dense(1, activation='linear',kernel_initializer='lecun_uniform',name='pre_pred_E', use_bias=False)(x)
    predictE = Clip(0.01,2.1)(predictE)
    predictE=Multiply(250,name='pred_E')(predictE)
    predictID=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    predictions = [predictID,predictE]
    
    model = Model(inputs=Inputs, outputs=predictions)
    return model



def resnet_like_3D_2(Inputs,nclasses,nregressions,dropoutRate=0.05,momentum=0.9, nodemulti=32, l2_lambda=0.0001):
    
    fullcaloimage = create_full_calo_image(Inputs, dropoutRate=0.0001, momentum=momentum)
    allayerenergies = create_per_layer_energies(Inputs)
    
    x = fullcaloimage
    
    
    
    #
    #repeat-vector can be concated at any per-layer step

    x_conv1 = Conv3D(int(0.5*nodemulti),kernel_size=(2, 2, 1),strides=(2,2,1), name='conv1_lin',
                      kernel_initializer='lecun_uniform',
                      kernel_regularizer=l2(0.2*l2_lambda),
                      border_mode='valid')(x)
    x_conv1 = LeakyReLU(alpha=leaky_relu_alpha)(x_conv1)   
    
    x_conv1_rn = Conv3D(nodemulti,kernel_size=(1, 1, 5),strides=(1,1,1), name='conv1_resnet_a',
                      kernel_initializer='lecun_uniform',
                      kernel_regularizer=l2(5*l2_lambda),
                      activation='relu',
                      trainable=False,
                      border_mode='same')(x)
    x_conv1_rn = Dropout(dropoutRate)(x_conv1_rn)  
    x_conv1_rn = Conv3D(int(0.5*nodemulti),kernel_size=(2, 2, 1),strides=(2,2,1), name='conv1_resnet_b',
                      kernel_initializer='zeros',
                      kernel_regularizer=l2(5*l2_lambda),
                      activation='relu',
                      trainable=False,
                      border_mode='valid')(x_conv1_rn)
    x_conv1_rn = Dropout(dropoutRate)(x_conv1_rn)  
    
    x = Add()([x_conv1,x_conv1_rn])
    
    
    
    x_conv2 = Conv3D(nodemulti,kernel_size=(4, 4, 1),strides=(4,4,1), name='conv2_lin',
                      kernel_initializer='lecun_uniform',
                      kernel_regularizer=l2(l2_lambda),
                      border_mode='valid')(x)
    x_conv2 = LeakyReLU(alpha=leaky_relu_alpha)(x_conv2)   
    
    x_conv2_rn = Conv3D(2*nodemulti,kernel_size=(1, 1, 7),strides=(1,1,1), name='conv2_resnet_a',
                      kernel_initializer='lecun_uniform',
                      kernel_regularizer=l2(5*l2_lambda),
                      activation='relu',
                      trainable=False,
                      border_mode='same')(x)
    x_conv2_rn = Dropout(dropoutRate)(x_conv2_rn)  
    x_conv2_rn = Conv3D(2*nodemulti,kernel_size=(7, 7, 1),strides=(1,1,1), name='conv2_resnet_b',
                      kernel_initializer='lecun_uniform',
                      kernel_regularizer=l2(5*l2_lambda),
                      activation='relu',
                      trainable=False,
                      border_mode='same')(x_conv2_rn)
    x_conv2_rn = Dropout(dropoutRate)(x_conv2_rn)  
    x_conv2_rn = Conv3D(nodemulti,kernel_size=(4, 4, 1),strides=(4,4,1), name='conv2_resnet_c',
                      kernel_initializer='zeros',
                      kernel_regularizer=l2(5*l2_lambda),
                      activation='relu',
                      trainable=False,
                      border_mode='valid')(x_conv2_rn)
    x_conv2_rn = Dropout(dropoutRate)(x_conv2_rn)  
  
    x = Add()([x_conv2,x_conv2_rn])
    
    print(allayerenergies.shape)
    allayerenergies = RepeatVector(4)(allayerenergies)
    print(allayerenergies.shape)
    allayerenergies = Reshape((2,2,18,1))(allayerenergies)
    x = Concatenate()([x,allayerenergies])
    
    x_conv3 = Conv3D(nodemulti,kernel_size=(1, 1, 4),strides=(1,1,4), name='conv3_lin',
                      kernel_initializer='lecun_uniform',
                      kernel_regularizer=l2(0.05*l2_lambda),
                      border_mode='valid')(x)
    x_conv3 = LeakyReLU(alpha=leaky_relu_alpha)(x_conv3)   
    
    x_conv3_rn = Conv3D(2*nodemulti,kernel_size=(1, 1, 7),strides=(1,1,1), name='conv3_resnet_a',
                      kernel_initializer='lecun_uniform',
                      kernel_regularizer=l2(5*l2_lambda),
                      activation='relu',
                      trainable=False,
                      border_mode='same')(x)
    x_conv3_rn = Dropout(dropoutRate)(x_conv3_rn)  
    x_conv3_rn = Conv3D(nodemulti,kernel_size=(1, 1, 4),strides=(1,1,4), name='conv3_resnet_b',
                      kernel_initializer='zeros',
                      kernel_regularizer=l2(5*l2_lambda),
                      activation='relu',
                      trainable=False,
                      border_mode='valid')(x_conv3_rn)
    x_conv3_rn = Dropout(dropoutRate)(x_conv3_rn)  
    x = Add()([x_conv3,x_conv3_rn])
    #up to here this is per layer
    
     
    
    x = Conv3D(2*nodemulti,kernel_size=(2, 2, 2),strides=(2,2,2), name='conv4_lin',
                      kernel_initializer=keras.initializers.random_normal(0, 0.1),
                      kernel_regularizer=l2(0.05*l2_lambda),
                      border_mode='valid')(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)   
    #x = BatchNormalization(momentum=momentum,name="norm_after_conv_block",
    #                                   center=True,scale=True)(x)
    x = Dropout(0.1*dropoutRate)(x) 
    
    
   
    x = Flatten()(x)
    
    #x = Dropout(dropoutRate)(x)
    
    x = Dense(32, kernel_initializer='lecun_uniform',name='dense3',activation='relu')(x)
    x = Dropout(0.1*dropoutRate)(x) 
    x = Dense(32, kernel_initializer='lecun_uniform',name='dense4',activation='relu')(x)
    
    predictE=Dense(1, activation='linear',kernel_initializer='lecun_uniform',name='pre_pred_E')(x)
    predictE = Clip(0.001,2.1)(predictE)
    predictE=Multiply(500,name='pred_E')(predictE)
    predictID=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    predictions = [predictID,predictE]
    
    model = Model(inputs=Inputs, outputs=predictions)
    return model


#also dows all the parsing
train=training_base(testrun=False,resumeSilently=False,renewtokens=True)


verbose=1
additionalplots=['pred_E_acc_calo_relative_rms',
                 'val_pred_E_acc_calo_relative_rms',
                 'pred_E_acc_calo_relative_rms_50',
                 'pred_E_acc_calo_relative_rms_100',
                 'pred_E_acc_calo_relative_rms_500',
                 'pred_E_acc_calo_relative_rms_1000',
                 'val_pred_E_acc_calo_relative_rms_50',
                 'val_pred_E_acc_calo_relative_rms_100',
                 'val_pred_E_acc_calo_relative_rms_500',
                 'val_pred_E_acc_calo_relative_rms_1000',
                 'val_pred_E_loss']

from Losses import huber_loss_calo, acc_calo_relative_rms, huber_loss_relative, huber_loss,low_value_msq,acc_calo_relative_rms_50,acc_calo_relative_rms_100,acc_calo_relative_rms_500,acc_calo_relative_rms_1000
usemetrics=['mean_squared_error',
            acc_calo_relative_rms,
            acc_calo_relative_rms_50,
            acc_calo_relative_rms_100,
            acc_calo_relative_rms_500,
            acc_calo_relative_rms_1000]
usefinetuneloss=huber_loss_calo #huber_loss_calo #low_value_msq #'mean_squared_error' # huber_loss_calo #huber_loss_relative #'mean_squared_error' #huber_loss_relative #huber_loss_relative #low_value_msq


batchsize=800
clipnorm=None

if not train.modelSet():
    
    nodemulti=16

    train.setModel(resnet_like_3D,dropoutRate=0.05,momentum=0.3,nodemulti=nodemulti,l2_lambda=1e-6)
    
    train.compileModel(learningrate=0.01,#will be overwritten anyway
                       clipnorm=1,
                   loss=['categorical_crossentropy','mean_squared_error'],metrics=usemetrics,
                   loss_weights=[0., 1.])

print(train.keras_model.summary())
#exit()


def open_resnet(train):
    print('un-freezing non-linear corrections, freezing dumb parts')
    #unfreeze etc
    for layer in train.keras_model.layers:
        if "res" in layer.name:
            layer.trainable=True
        if "dumb" in layer.name:
            layer.trainable=False
    freeze_all_batchnorms(train.keras_model)
    train.compileModel(learningrate=0.01, #anyway overwritten
               clipnorm=clipnorm,
               loss=['categorical_crossentropy',usefinetuneloss],#'mean_squared_error'],#'mean_squared_error'],#huber_loss],#mean_squared_logarithmic_error],
               metrics=usemetrics,
               loss_weights=[0., 1.])
    print(train.keras_model.summary())
    
def open_conv_lin(train):
    print('un-freezing linear corrections')
    #unfreeze etc
    for layer in train.keras_model.layers:
        if "lin" in layer.name:
            layer.trainable=True
    train.compileModel(learningrate=0.01, #anyway overwritten
               clipnorm=clipnorm,
               loss=['categorical_crossentropy',usefinetuneloss],#'mean_squared_error'],#'mean_squared_error'],#huber_loss],#mean_squared_logarithmic_error],
               metrics=usemetrics,
               loss_weights=[0., 1.])
    print(train.keras_model.summary())
    
def open_conv_lsum(train):
    print('un-freezing lsum')
    #unfreeze etc
    for layer in train.keras_model.layers:
        if "lsum" in layer.name:
            layer.trainable=True
    train.compileModel(learningrate=0.01, #anyway overwritten
               clipnorm=1,
               loss=['categorical_crossentropy','mean_squared_error'],#'mean_squared_error'],#'mean_squared_error'],#huber_loss],#mean_squared_logarithmic_error],
               metrics=usemetrics,
               loss_weights=[0., 1.])
    print(train.keras_model.summary())
    
def open_pre(train):
    print('un-freezing pre')
    #unfreeze etc
    for layer in train.keras_model.layers:
        if "pre_" in layer.name:
            layer.trainable=True
    train.compileModel(learningrate=0.01, #anyway overwritten
               clipnorm=1,
               loss=['categorical_crossentropy','mean_squared_error'],#'mean_squared_error'],#'mean_squared_error'],#huber_loss],#mean_squared_logarithmic_error],
               metrics=usemetrics,
               loss_weights=[0., 1.])
    print(train.keras_model.summary())
    

def freeze_batchnorm(train):
    print('freezing bath norm')
    freeze_all_batchnorms(train.keras_model)   
    
def exit_func(train):
    exit()

import collections
Learning_sched = collections.namedtuple('Learning_sched', 'lr nepochs batchmulti funccall')
learn=[]

#K.set_value(train.keras_model.optimizer.lr, lrs.lr)
learnmulti=1
learn.append(Learning_sched(lr=learnmulti*1e-3,     nepochs=1, batchmulti=0.25,  funccall=None))
learn.append(Learning_sched(lr=learnmulti*5e-5,     nepochs=10, batchmulti=0.5,    funccall=None))
learn.append(Learning_sched(lr=learnmulti*1e-5,     nepochs=20, batchmulti=1,  funccall=open_conv_lin))
learn.append(Learning_sched(lr=learnmulti*1e-5 ,    nepochs=20,  batchmulti=1, funccall=open_conv_lin))
learn.append(Learning_sched(lr=learnmulti*1e-5,     nepochs=60,  batchmulti=1,   funccall=open_resnet))
#learn.append(Learning_sched(lr=learnmulti*1e-5,     nepochs=1,  batchmulti=1,   funccall=freeze_batchnorm))

totalepochs=0
import keras.backend as K
for lrs in learn:

    if train.trainedepoches<=totalepochs:
        if lrs.funccall:
            lrs.funccall(train)
        K.set_value(train.keras_model.optimizer.lr, lrs.lr)
        print('set learning rate to '+str(lrs.lr))
        model,history = train.trainModel(nepochs=totalepochs+lrs.nepochs, 
                                 batchsize=int(batchsize*lrs.batchmulti),
                                 checkperiod=1,
                                 verbose=verbose,
                                 additional_plots=additionalplots)
        totalepochs+=lrs.nepochs
    else:
        print('skipping already trained epochs: '+str(train.trainedepoches))
        if lrs.nepochs+totalepochs<train.trainedepoches:
            totalepochs+=lrs.nepochs
        else:
            totalepochs=train.trainedepoches
    







