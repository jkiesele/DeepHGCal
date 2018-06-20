

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
l2_lambda = 0.005


############# just a placeholder

def sofiamodel(Inputs,nclasses,nregressions,dropoutRate=0.05,momentum=0.6):
    #image = Input(shape=(25, 25, 25, 1))
    
    regulariser=l2(l2_lambda)
 
    #34x34x8x2
    ecalhits= BatchNormalization(momentum=momentum,center=False)(Inputs[0])
    ecalhits = Dropout(dropoutRate)(ecalhits)
    
    #17x17x10x2
    hcalhits= BatchNormalization(momentum=momentum,center=False)(Inputs[1])
    hcalhits = Dropout(dropoutRate)(hcalhits)
    
    #output: 17x17x8x32
    ecalcompressed = Convolution3D(8,kernel_size=(5,5,5),strides=(2,2,1), activation='relu',
                      kernel_initializer='lecun_uniform',kernel_regularizer=regulariser,
                      border_mode='same',name='ecalcompressed')(ecalhits) 
                      
    hcalpreprocessed = Convolution3D(8,kernel_size=(5,5,5),strides=(1,1,1), activation='relu',
                      kernel_initializer='lecun_uniform',kernel_regularizer=regulariser,
                      border_mode='same',name='hcalpreprocessed')(hcalhits) 
    
    #merge to the full calo
    fullcaloimage = Concatenate(axis=-2,name='fullcaloimage')([ecalcompressed,hcalpreprocessed])
    fullcaloimage = BatchNormalization(momentum=momentum)(fullcaloimage)
    fullcaloimage = Dropout(dropoutRate)(fullcaloimage)
    
    #non-muon ID part
    
    x = Convolution3D(24,kernel_size=(5, 5, 5),strides=(1,1,1), activation='relu',name='conv1',
                      kernel_initializer='lecun_uniform',kernel_regularizer=regulariser,
                      border_mode='same')(fullcaloimage)
    x = BatchNormalization(momentum=momentum)(x)
    x = Dropout(dropoutRate)(x)
    
    x = Convolution3D(12,kernel_size=(5, 5, 5),strides=(2,2,2), activation='relu',name='conv2',
                      kernel_initializer='lecun_uniform',kernel_regularizer=regulariser,
                      border_mode='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Dropout(dropoutRate)(x)
    
    x = Convolution3D(16,kernel_size=(5, 5, 5),strides=(2,2,2), activation='relu',name='conv3',
                      kernel_initializer='lecun_uniform',kernel_regularizer=regulariser,
                      border_mode='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Dropout(dropoutRate)(x)
    
    x = Convolution3D(24,kernel_size=(3, 3, 3),strides=(2,2,2), activation='relu',name='conv4',
                      kernel_initializer='lecun_uniform',kernel_regularizer=regulariser,
                      border_mode='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Dropout(dropoutRate)(x)
    
    x = Convolution3D(24,kernel_size=(3, 3, 3),strides=(2,2,2), activation='relu',name='conv5',
                      kernel_initializer='lecun_uniform',kernel_regularizer=regulariser,
                      border_mode='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Dropout(dropoutRate)(x)
    
    x = Flatten()(x)
    
    ## muon dedicated part
    smallshower = Cropping3D(cropping=((5,5),(5,5),(0,0)),name='muoncrop')(fullcaloimage)
    smallshower=Convolution3D(16, (5, 5, 3),strides=(1,1,1) , border_mode='valid',activation='relu',
                                  kernel_regularizer=regulariser,name='muon0' )(smallshower)
    smallshower = Dropout(dropoutRate)(smallshower)   
    smallshower=Convolution3D(16, (3, 3, 5),strides=(1,1,3) , border_mode='valid',activation='relu',
                                  kernel_regularizer=regulariser,name='muon1' )(smallshower)
    smallshower = BatchNormalization(momentum=momentum)(smallshower)
    smallshower = Dropout(dropoutRate)(smallshower)   
    smallshower = Flatten()(smallshower)
    
    x = Concatenate()([x,smallshower])
    x = Dropout(dropoutRate)(x) #double dropout here
    
    x = Dense(128, activation='relu',kernel_initializer='lecun_uniform',name='firstDense')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(32, activation='relu',kernel_initializer='lecun_uniform',name='dense2')(x) 
    x = Dropout(dropoutRate)(x)
    
    predictE=Dense(1, activation='linear',kernel_initializer='lecun_uniform',name='pred_E')(x)
    predictID=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    predictions = [predictID,predictE]
    
    model = Model(inputs=Inputs, outputs=predictions)
    return model


#also dows all the parsing
train=training_base(testrun=False,resumeSilently=True)


if not train.modelSet():
    from Losses import loss_modRelMeanSquaredError

    train.setModel(sofiamodel,dropoutRate=0.15,momentum=0.6)
    train.compileModel(learningrate=0.002,
                   loss=['categorical_crossentropy',loss_modRelMeanSquaredError],#mean_squared_logarithmic_error],
                   metrics=['accuracy'],
                   loss_weights=[1., 0.])
    
    model,history = train.trainModel(nepochs=2, 
                                 batchsize=1000, 
                                 maxqsize=7,
                                 checkperiod=2,
                                 verbose=1)
    
    train.compileModel(learningrate=0.0002,
                   loss=['categorical_crossentropy',loss_modRelMeanSquaredError],#mean_squared_logarithmic_error],
                   metrics=['accuracy'],
                   loss_weights=[1., 0.])
    
    
    model,history = train.trainModel(nepochs=8, 
                                 batchsize=1000,
                                 maxqsize=7,
                                 checkperiod=2,
                                 verbose=1)
    
    train.compileModel(learningrate=0.00002,
                   loss=['categorical_crossentropy',loss_modRelMeanSquaredError],#mean_squared_logarithmic_error],
                   metrics=['accuracy'],
                   loss_weights=[1., 0.])
    

print(train.keras_model.summary())


#exit()
model,history = train.trainModel(nepochs=160, 
                                 batchsize=1000, 
                                 lr_factor=0.3, 
                                 lr_patience=5, 
                                 lr_epsilon=0.001, 
                                 lr_cooldown=5, 
                                 lr_minimum=0.00000002, 
                                 maxqsize=7,
                                 checkperiod=2,
                                 verbose=1)
