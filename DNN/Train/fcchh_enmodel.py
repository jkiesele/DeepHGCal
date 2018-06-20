

from DeepJetCore.training.training_base import training_base
import keras
from keras.layers import Dense, ZeroPadding3D,Conv3D,Dropout, Flatten, Convolution2D,Convolution3D, merge, Convolution1D, Conv2D, LSTM, LocallyConnected2D
from keras.layers.pooling import MaxPooling3D
from keras.models import Model
from keras.layers.core import Reshape, Masking, Permute
from keras.layers.pooling import MaxPooling2D, MaxPooling3D,AveragePooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate,Add,Multiply
from keras.layers.noise import GaussianDropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Cropping3D
from keras.regularizers import l2
l2_lambda = 0.0005

from Layers import Sum3DFeatureOne,Sum3DFeaturePerLayer
############# just a placeholder

def sofiamodel(Inputs,nclasses,nregressions,dropoutRate=0.05,momentum=0.6):
    #image = Input(shape=(25, 25, 25, 1))
    
    regulariser=l2(l2_lambda)
 
    #34x34x8x2
    ecalhits= Inputs[0] #BatchNormalization(momentum=momentum,center=False)(Inputs[0])
    ecalhits = Dropout(dropoutRate)(ecalhits)
    
    #17x17x10x2
    hcalhits= Inputs[1]# BatchNormalization(momentum=momentum,center=False)(Inputs[1])
    hcalhits = Dropout(dropoutRate)(hcalhits)
    
    ### deep per-hit calibration
    ecalhits       = Convolution3D(8,kernel_size=(1,1,1),strides=(1,1,1), activation='relu',
                      kernel_initializer='lecun_uniform',
                      border_mode='same',name='ecalhits1')(ecalhits)             
    ecalcompressed = Convolution3D(8,kernel_size=(2,2,1),strides=(2,2,1), activation='relu',
                      kernel_initializer='lecun_uniform',
                      border_mode='same',name='ecalhits5')(ecalhits) 
                      
    
    hcalhits  = Convolution3D(8,kernel_size=(1,1,1),strides=(1,1,1), activation='relu',
                      kernel_initializer='lecun_uniform',
                      border_mode='same',name='hcalhits5')(hcalhits)  
    
    #merge to the full calo
    fullcaloimage = Concatenate(axis=-2,name='fullcaloimage')([ecalcompressed,hcalhits])
    fullcaloimage = BatchNormalization(momentum=momentum,center=False)(fullcaloimage)
    fullcaloimage = Dropout(dropoutRate)(fullcaloimage)
    #now just basically sum in layers and then in depth
    #17x17x18x8
    
    x = Convolution3D(32,kernel_size=(5, 5, 5),strides=(1,1,1), activation='relu',name='conv1',
                      kernel_initializer='lecun_uniform',
                      border_mode='same')(fullcaloimage)
    x = Dropout(3*dropoutRate)(x)
    
    x = Convolution3D(32,kernel_size=(5, 5, 5),strides=(1,1,1), activation='relu',name='conv2',
                      kernel_initializer='lecun_uniform',
                      border_mode='same')(x)
    x = Dropout(3*dropoutRate)(x)
    
    x = Convolution3D(32,kernel_size=(5, 5, 5),strides=(3,3,1), activation='relu',name='conv3',
                      kernel_initializer='lecun_uniform',
                      border_mode='same')(x)
    x = Dropout(3*dropoutRate)(x)
    
    x = Convolution3D(16,kernel_size=(3, 3, 6),strides=(3,3,1), activation='relu',name='conv4',
                      kernel_initializer='lecun_uniform',
                      border_mode='same')(x)
    x = BatchNormalization(momentum=momentum,center=False)(x)
    x = Dropout(3*dropoutRate)(x)
    #2x2x18x4
    
    x = Flatten()(x)
    x = Dense(32, activation='relu',kernel_initializer='zeros',name='dense0',trainable=False)(x) 
    x = Dropout(3*dropoutRate)(x)
    
    #totalecalenergy=Sum3DFeatureOne()(Inputs[0])
    #totalhcalenergy=Sum3DFeatureOne()(Inputs[1])
    #alltotalenergy=Add()([totalecalenergy,totalhcalenergy])
    
    perlayerenergies=[]
    for i in range(8):
        perlayerenergies.append(Sum3DFeaturePerLayer(i)(Inputs[0]))
    for i in range(10):
        perlayerenergies.append(Sum3DFeaturePerLayer(i)(Inputs[1]))
    
    allayerenergies=Concatenate()(perlayerenergies)
    x=Concatenate()([x,allayerenergies])
    
    
    x = Dense(64, activation='relu',kernel_initializer='lecun_uniform',name='dense1')(x) 
    x = Dropout(dropoutRate)(x)
    x = Dense(64, activation='relu',kernel_initializer='zeros',name='dense2',trainable=False)(x) 
    x = Dropout(dropoutRate)(x)
    x=Concatenate()([x,allayerenergies])
    x = Dense(32, activation='relu',kernel_initializer='lecun_uniform',name='dense3')(x) 
    x = Dropout(0.2*dropoutRate)(x)
    
    predictE=Dense(1, activation='linear',kernel_initializer='lecun_uniform',name='pred_E')(x)
    predictID=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    predictions = [predictID,predictE]
    
    model = Model(inputs=Inputs, outputs=predictions)
    return model


#also dows all the parsing
train=training_base(testrun=False,resumeSilently=False)

batchsize=1000
verbose=1

from Losses import huber_loss_calo, acc_calo_relative_rms
if not train.modelSet():
    
    usemetrics=['mean_squared_error',acc_calo_relative_rms]
    usefinetuneloss=huber_loss_calo

    train.setModel(sofiamodel,dropoutRate=0.05,momentum=0.6)
    
    train.compileModel(learningrate=0.1,
                   loss=['categorical_crossentropy',usefinetuneloss],#mean_squared_logarithmic_error],
                   metrics=usemetrics,
                   loss_weights=[0., 1.])
    print(train.keras_model.summary())
    
    #first learn total energy, trivial, no room for overtraining
    model,history = train.trainModel(nepochs=1, 
                                 batchsize=batchsize, 
                                 maxqsize=7,
                                 checkperiod=1,
                                 verbose=verbose)
    
    train.compileModel(learningrate=0.01,
                   loss=['categorical_crossentropy',usefinetuneloss],#mean_squared_logarithmic_error],
                   metrics=usemetrics,
                   loss_weights=[0., 1])
    
    model,history = train.trainModel(nepochs=1, 
                                 batchsize=batchsize, 
                                 maxqsize=7,
                                 checkperiod=1,
                                 verbose=verbose)
    
    
    train.compileModel(learningrate=0.001,
                   loss=['categorical_crossentropy',usefinetuneloss],#mean_squared_logarithmic_error],
                   metrics=usemetrics,
                   loss_weights=[0., 1])
    
    model,history = train.trainModel(nepochs=4, 
                                 batchsize=batchsize, 
                                 maxqsize=7,
                                 checkperiod=1,
                                 verbose=verbose)
    
    print('opening non-linear corrections to layer energies')
    for layer in model.layers:
        if layer.name == 'dense2':
            layer.trainable=True
    
    train.compileModel(learningrate=0.0005,
                   loss=['categorical_crossentropy',usefinetuneloss],#mean_squared_logarithmic_error],
                   metrics=usemetrics,
                   loss_weights=[0., 1])
    
    print(train.keras_model.summary())
    
    model,history = train.trainModel(nepochs=10, 
                                 batchsize=batchsize, 
                                 maxqsize=7,
                                 checkperiod=1,
                                 verbose=verbose)
    
    
    
    print('opening shower-shape corrections to layer energies')
    for layer in model.layers:
        layer.trainable=True
    
    train.compileModel(learningrate=0.0005,
                   loss=['categorical_crossentropy',usefinetuneloss],#mean_squared_logarithmic_error],
                   metrics=usemetrics,
                   loss_weights=[0., 1])
    
    print(train.keras_model.summary())
    
    model,history = train.trainModel(nepochs=5, 
                                 batchsize=batchsize, 
                                 maxqsize=7,
                                 checkperiod=2,
                                 verbose=verbose)
    
    train.compileModel(learningrate=0.00001,
                   loss=['categorical_crossentropy',usefinetuneloss],#mean_squared_logarithmic_error],
                   metrics=usemetrics,
                   loss_weights=[0., 1])
    

print(train.keras_model.summary())


#exit()
model,history = train.trainModel(nepochs=100, 
                                 batchsize=batchsize, 
                                 stop_patience=300, 
                                 lr_factor=0.5, 
                                 lr_patience=10, 
                                 lr_epsilon=0.001, 
                                 lr_cooldown=10, 
                                 lr_minimum=0.0000001, 
                                 maxqsize=7,
                                 checkperiod=2,
                                 verbose=verbose)
