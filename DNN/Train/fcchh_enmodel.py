

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

from Layers import Sum3DFeatureOne,Sum3DFeaturePerLayer, SelectEnergyOnly, ReshapeBatch, Multiply, Log_plus_one, Clip, Print, Reduce_sum, Sum_difference_squared
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


def resnet_like_3D(Inputs,nclasses,nregressions,dropoutRate=0.05,momentum=0.6, nodemulti=32, l2_lambda=0.0001, use_dumb_bias=False):
    
    fullcaloimage = create_full_calo_image(Inputs, dropoutRate=0.02, momentum=-1,trainable=True)
    
    
    allayers = create_per_layer_energies(Inputs)
    
    print(allayers.shape)
    
    x = fullcaloimage

    #make the really dumb part that gets corrections
    
    
    x = create_conv_resnet(x, name='conv_block_1',
                       kernel_dumb=(2,2,1), strides_dumb=(2,2,1),
                       nodes_lin=16,
                       nodes_nonlin=24, kernel_nonlin_a=(3,3,2), kernel_nonlin_b=(2,2,3), lambda_reg=l2_lambda,
                       leaky_relu_alpha=0.01,
                       dropout=dropoutRate,
                       lin_trainable=True,
                       use_dumb_bias=use_dumb_bias)
    
    diffA = Sum_difference_squared()([allayers,x])
    #diffA=Print("diffA")(diffA)
    
    #x = Dropout(0.1*dropoutRate)(x)
    
    x = create_conv_resnet(x, name='conv_block_2',
                       kernel_dumb=(4,4,1), strides_dumb=(4,4,1),
                       nodes_lin=16,
                       nodes_nonlin=16, kernel_nonlin_a=(4,4,2), kernel_nonlin_b=(2,2,3), lambda_reg=l2_lambda,
                       leaky_relu_alpha=0.01, 
                       dropout=dropoutRate,
                       lin_trainable=True,
                       use_dumb_bias=use_dumb_bias)
    
    diffB = Sum_difference_squared()([allayers,x])
    #x = Dropout(0.1*dropoutRate)(x)
    
    x = create_conv_resnet(x, name='conv_block_3',
                       kernel_dumb=(1,1,3), strides_dumb=(1,1,3),
                       nodes_lin=16,
                       nodes_nonlin=16, kernel_nonlin_a=(3,3,2), kernel_nonlin_b=(2,2,4), lambda_reg=l2_lambda,
                       leaky_relu_alpha=0.01,
                       dropout=dropoutRate,
                       lin_trainable=True,
                       use_dumb_bias=use_dumb_bias)
    diffC = Sum_difference_squared()([allayers,x])
    
    #x = Dropout(0.1*dropoutRate)(x)
    
    
    x = create_conv_resnet(x, name='conv_block_4',
                       kernel_dumb=(2,2,3), strides_dumb=(2,2,3),
                       nodes_lin=16,
                       nodes_nonlin=16, kernel_nonlin_a=(3,3,3), kernel_nonlin_b=(3,3,3), lambda_reg=l2_lambda,
                       leaky_relu_alpha=0.01,
                       dropout=dropoutRate,
                       lin_trainable=True,
                       normalize_dumb=False,
                       nodes_dumb=1,dumb_trainable=False,
                       use_dumb_bias=use_dumb_bias)
    
    
    diffD = Sum_difference_squared()([allayers,x])
    
    xsum = Sum3DFeatureOne(0)(x)
    #x = Dropout(0.1*dropoutRate)(x)
    #x = Print("x")(x)
    x = Flatten()(x)
    
    
    alldiff = Add()([diffA,diffB,diffC,diffD])
    
    
    x = Dense(32, kernel_initializer=keras.initializers.random_normal(1./32., 1./32.),name='dense1', trainable=True)(x)
    x = LeakyReLU(alpha=0.001)(x)
    x = Dense(12, kernel_initializer=keras.initializers.random_normal(1./(32.*120.), 1./(32.*120.)),name='dense2', trainable=True)(x)
    
    
   
    predictE=Dense(1, activation='linear',kernel_initializer=keras.initializers.random_normal(1./12., 1./12.),name='pre_pred_E')(x)
    
    predictE=Multiply(200.)(predictE)
    #predictE=Print("predictE")(predictE)
    predictE = Clip(-1000,2000,name='pred_E')(predictE)
    #predictE=Print("predictE")(predictE)
    
    #alldiff=Multiply(1e5)(alldiff)
    #alldiff=Print("alldiff")(alldiff)
    predictID=RepeatVector(nclasses)(alldiff)
    predictID=Reshape([nclasses],name="ID_pred")(predictID)
    predictions = [predictID,predictE]
    
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
DataCollection.generator=_adapted_generator

#orig2=train.val_data.generator
#def _adapted_generator2(self):
#    for t in orig2(self):
#        x=t[0]
#        y=t[1]
#        y[0]*=0
#        yield (x,y)
#train.val_data.generator=_adapted_generator2

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
clipnorm=1

if not train.modelSet():
    
    nodemulti=16

    train.setModel(resnet_like_3D,dropoutRate=0.1,momentum=0.3,nodemulti=nodemulti,l2_lambda=1e-4)
    
    train.compileModel(learningrate=0.01,#will be overwritten anyway
                       clipnorm=1,
                   loss=['mean_absolute_error','mean_squared_error'],metrics=usemetrics,
                   loss_weights=[1., 1e-7]) #ID, en

#print(train.keras_model.summary())
#exit()


    
def open_en_training(train):
    print('open energy loss and dense layers')
    for l in train.keras_model.layers:
        if "dense" in l.name:
            l.trainable=True
            l.use_bias=True
        if "pred_E" == l.name: #clip layer
            l.min=-1000
                    
    
def open_conv_lin(train):
    print('un-freezing linear corrections')
    #unfreeze etc
    for layer in train.keras_model.layers:
        if "lin" in layer.name:
            layer.trainable=True
        if "pred_E" == layer.name: #clip layer
            layer.min=1
    
    
def open_pre(train):
    print('un-freezing pre')
    #unfreeze etc
    for layer in train.keras_model.layers:
        if "pre_" in layer.name:
            layer.trainable=True
    

def freeze_batchnorm(train):
    print('freezing bath norm')
    freeze_all_batchnorms(train.keras_model)   
    
def exit_func(train):
    train.saveModel("pretrained.h5")
    print('exit pre-training')
    exit()
    
    
    
def open_resnet(train):
    print('un-freezing non-linear corrections, freezing dumb parts')
    #unfreeze etc
    for layer in train.keras_model.layers:
        if "res" in layer.name:
            layer.trainable=True
        if "dumb" in layer.name:
            layer.trainable=True
        if "pred_E" == layer.name: #clip layer
            layer.min=1
    print(train.keras_model.summary())

import collections
Learning_sched = collections.namedtuple('Learning_sched', 'lr nepochs batchmulti funccall loss_weights en_loss')
learn=[]

train.trainedepoches=0
#K.set_value(train.keras_model.optimizer.lr, lrs.lr)


verbose=1

#pre-train the first layers
#learn.append(Learning_sched(lr=1e-3,     nepochs=3,   batchmulti=0.5,  funccall=None,         loss_weights=[1e-7, 1], en_loss='mean_squared_error'))
#learn.append(Learning_sched(lr=1e-4,     nepochs=4,   batchmulti=1,    funccall=None,         loss_weights=None, en_loss=None))
#learn.append(Learning_sched(lr=1e-6,     nepochs=4,   batchmulti=1,    funccall=None,         loss_weights=[1e-7, 100.], en_loss=huber_loss_calo))
#learn.append(Learning_sched(lr=1e-6,     nepochs=15,   batchmulti=1,   funccall=open_resnet,  loss_weights=[1e-7, 100.], en_loss=huber_loss_calo))
#learn.append(Learning_sched(lr=1e-7,     nepochs=25,   batchmulti=1,   funccall=None,         loss_weights=None, en_loss=None))


learn.append(Learning_sched(lr=1e-3,     nepochs=3,   batchmulti=0.5,  funccall=open_resnet,         loss_weights=[1e-7, 100], en_loss=huber_loss_calo))
learn.append(Learning_sched(lr=1e-4,     nepochs=4,   batchmulti=1,    funccall=None,         loss_weights=None, en_loss=None))
learn.append(Learning_sched(lr=1e-5,     nepochs=4,   batchmulti=1,    funccall=None,         loss_weights=None, en_loss=None))
learn.append(Learning_sched(lr=1e-6,     nepochs=15,   batchmulti=1,    funccall=None,         loss_weights=None, en_loss=None))
learn.append(Learning_sched(lr=1e-7,     nepochs=25,   batchmulti=1,    funccall=None,         loss_weights=None, en_loss=None))

totalepochs=0
import keras.backend as K
for lrs in learn:

    if train.trainedepoches<=totalepochs:
        if lrs.funccall:
            lrs.funccall(train)
        if lrs.loss_weights:
            train.compileModel(learningrate=0.01, #anyway overwritten
               clipnorm=clipnorm,
               loss=['mean_absolute_error',lrs.en_loss],#'mean_squared_error'],#'mean_squared_error'],#huber_loss],#mean_squared_logarithmic_error],
               metrics=usemetrics,
               loss_weights=lrs.loss_weights)
            print(train.keras_model.summary())
        
        for l in train.keras_model.layers:
            if l.trainable and l.weights and len(l.weights):
                print('trainable '+l.name)
                
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
    







