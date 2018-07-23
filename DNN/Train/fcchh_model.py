

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
from Layers import PermuteBatch, ReshapeBatch, SelectEnergyOnly
l2_lambda = 0.0005

batch_size = 800
def shahrukh_model(Inputs,nclasses,nregressions,dropoutRate=0.05,momentum=0.6):
    global batch_size
    # image = Input(shape=(25, 25, 25, 1))

    #34x34x8x2
    ecalhits= SelectEnergyOnly()(Inputs[0])
    ecalhits= BatchNormalization(momentum=momentum,center=False)(ecalhits)
    ecalhits = Dropout(dropoutRate)(ecalhits)
    
    #17x17x10x2
    hcalhits= SelectEnergyOnly()(Inputs[1])
    hcalhits= BatchNormalization(momentum=momentum,center=False)(hcalhits)
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
    
    x = fullcaloimage

    print("Before", K.int_shape(x))

    # Shape: [B,13,13,L,C]
    shape = K.int_shape(x)
    _, _, _, L, C = shape



    x = Conv3D(30, kernel_size=(5, 5, 1), padding='same', activation='relu',
                      kernel_initializer='lecun_uniform', kernel_regularizer=l2(l2_lambda))(x)
    x = Conv3D(30, kernel_size=(1, 1, 5), padding='same', activation='relu',
                      kernel_initializer='lecun_uniform', kernel_regularizer=l2(l2_lambda))(x)
    x = Conv3D(30, kernel_size=(5, 5, 1), padding='same', activation='relu',
                      kernel_initializer='lecun_uniform', kernel_regularizer=l2(l2_lambda))(x)
    x = Conv3D(30, kernel_size=(1, 1, 5), padding='same', activation='relu',
                      kernel_initializer='lecun_uniform', kernel_regularizer=l2(l2_lambda))(x)
    x = Conv3D(30, kernel_size=(5, 5, 1), padding='same', activation='relu',
                      kernel_initializer='lecun_uniform', kernel_regularizer=l2(l2_lambda))(x)
    x = Conv3D(30, kernel_size=(1, 1, 5), padding='same', activation='relu',
                      kernel_initializer='lecun_uniform', kernel_regularizer=l2(l2_lambda))(x)
    x = Conv3D(30, kernel_size=(5, 5, 1), padding='same', activation='relu',
                      kernel_initializer='lecun_uniform', kernel_regularizer=l2(l2_lambda))(x)
    x = Conv3D(30, kernel_size=(1, 1, 5), padding='same', activation='relu',
                      kernel_initializer='lecun_uniform', kernel_regularizer=l2(l2_lambda))(x)
    print(x.shape)
    x = ReshapeBatch((-1 , L , 30))(x)
    
    print(x.shape)
    
    lstm_1 = LSTM(100, return_sequences=True, input_shape=((13*13),L, 30))
    lstm_2 = LSTM(100, return_sequences=True, input_shape=((13*13),L, 100))

    x = lstm_1(x)
    x = lstm_2(x)
    
    x = LSTM(100, activation='relu')(x)

    x = Dropout(dropoutRate)(x) #double dropout here
    
    x = Dense(64, activation='relu',kernel_initializer='lecun_uniform',name='firstDense')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(32, activation='relu',kernel_initializer='lecun_uniform',name='dense2')(x) 
    x = Dropout(dropoutRate)(x)
    
    predictE=Dense(1, activation='linear',kernel_initializer='lecun_uniform',name='pred_E')(x)
    predictID=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    predictions = [predictID,predictE]
    
    model = Model(inputs=Inputs, outputs=predictions)
    return model
    



############# just a placeholder

def sofiamodel(Inputs,nclasses,nregressions,dropoutRate=0.05,momentum=0.6):
    #image = Input(shape=(25, 25, 25, 1))
    
    regulariser=l2(l2_lambda)
 
    #34x34x8x2
    ecalhits= Inputs[0]
    ecalhits= BatchNormalization(momentum=momentum,center=False)(ecalhits)
    ecalhits = Dropout(0.2*dropoutRate)(ecalhits)
    
    #17x17x10x2
    hcalhits= Inputs[1]
    hcalhits= BatchNormalization(momentum=momentum,center=False)(hcalhits)
    hcalhits = Dropout(0.2*dropoutRate)(hcalhits)
    
    #output: 17x17x8x32
    ecalhits = Convolution3D(32,kernel_size=(1,1,1),strides=(1,1,1), activation='relu',
                      kernel_initializer='lecun_uniform',kernel_regularizer=regulariser,
                      border_mode='same',name='ecalcompressed0')(ecalhits) 
    ecalcompressed = Convolution3D(16,kernel_size=(2,2,2),strides=(2,2,1), activation='relu',
                      kernel_initializer='lecun_uniform',kernel_regularizer=regulariser,
                      border_mode='same',name='ecalcompressed')(ecalhits) 
                      
    hcalpreprocessed = Convolution3D(16,kernel_size=(1,1,1),strides=(1,1,1), activation='relu',
                      kernel_initializer='lecun_uniform',kernel_regularizer=regulariser,
                      border_mode='same',name='hcalpreprocessed')(hcalhits) 
    
    #merge to the full calo
    fullcaloimage = Concatenate(axis=-2,name='fullcaloimage')([ecalcompressed,hcalpreprocessed])
    fullcaloimage = BatchNormalization(momentum=momentum)(fullcaloimage)
    fullcaloimage = Dropout(dropoutRate)(fullcaloimage)
    
    #non-muon ID part
    
    x = Convolution3D(64,kernel_size=(6, 6, 1),strides=(1,1,1), activation='relu',name='conv1',
                      kernel_initializer='lecun_uniform',kernel_regularizer=regulariser,
                      border_mode='same')(fullcaloimage)
    #x = BatchNormalization(momentum=momentum)(x)
    x = Dropout(dropoutRate)(x)
    x = Convolution3D(24,kernel_size=(1, 1, 5),strides=(1,1,1), activation='relu',name='conv1a',
                      kernel_initializer='lecun_uniform',kernel_regularizer=regulariser,
                      border_mode='same')(fullcaloimage)
    x = BatchNormalization(momentum=momentum)(x)
    x = Dropout(dropoutRate)(x)
    
    x = Convolution3D(48,kernel_size=(5, 5, 1),strides=(2,2,1), activation='relu',name='conv2',
                      kernel_initializer='lecun_uniform',kernel_regularizer=regulariser,
                      border_mode='same')(x)
    x = Convolution3D(32,kernel_size=(1, 1, 5),strides=(1,1,2), activation='relu',name='conv2a',
                      kernel_initializer='lecun_uniform',kernel_regularizer=regulariser,
                      border_mode='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Dropout(dropoutRate)(x)
    
    x = Convolution3D(48,kernel_size=(5, 5, 1),strides=(2,2,1), activation='relu',name='conv3',
                      kernel_initializer='lecun_uniform',kernel_regularizer=regulariser,
                      border_mode='same')(x)                
    x = Convolution3D(48,kernel_size=(1, 1, 5),strides=(1,1,2), activation='relu',name='conv3a',
                      kernel_initializer='lecun_uniform',kernel_regularizer=regulariser,
                      border_mode='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Dropout(dropoutRate)(x)
    
    x = Convolution3D(64,kernel_size=(3, 3, 1),strides=(2,2,1), activation='relu',name='conv4',
                      kernel_initializer='lecun_uniform',kernel_regularizer=regulariser,
                      border_mode='same')(x)
    x = Convolution3D(48,kernel_size=(1, 1, 3),strides=(1,1,2), activation='relu',name='conv4a',
                      kernel_initializer='lecun_uniform',kernel_regularizer=regulariser,
                      border_mode='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Dropout(dropoutRate)(x)
    
    x = Convolution3D(32,kernel_size=(3, 3, 3),strides=(2,2,2), activation='relu',name='conv5',
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

queue_size=5
additionalplots=None #['val_ID_pred_loss','ID_pred_loss']

if not train.modelSet():
    from Losses import loss_modRelMeanSquaredError

    train.setModel(sofiamodel,dropoutRate=0.15,momentum=0.6)
    train.compileModel(learningrate=0.002,
                   loss=['categorical_crossentropy',loss_modRelMeanSquaredError],#mean_squared_logarithmic_error],
                   metrics=['accuracy'],
                   loss_weights=[1., 0.])
    
    print(train.keras_model.summary())
    #exit()
    model,history = train.trainModel(nepochs=3, 
                                 batchsize=batch_size, 
                                 maxqsize=queue_size,
                                 checkperiod=2,
                                 verbose=1,
                                 additional_plots=additionalplots)
    
    train.compileModel(learningrate=0.0002,
                   loss=['categorical_crossentropy',loss_modRelMeanSquaredError],#mean_squared_logarithmic_error],
                   metrics=['accuracy'],
                   loss_weights=[1., 0.])
    
    
    model,history = train.trainModel(nepochs=8, 
                                 batchsize=batch_size,
                                 maxqsize=queue_size,
                                 checkperiod=2,
                                 verbose=1)
    
    train.compileModel(learningrate=0.00002,
                   loss=['categorical_crossentropy',loss_modRelMeanSquaredError],#mean_squared_logarithmic_error],
                   metrics=['accuracy'],
                   loss_weights=[1., 0.])
    

print(train.keras_model.summary())


#exit()
model,history = train.trainModel(nepochs=100, 
                                 batchsize=batch_size, 
                                 lr_factor=0.3, 
                                 lr_patience=10, 
                                 lr_epsilon=0.001, 
                                 lr_cooldown=10, 
                                 lr_minimum=0.00000002, 
                                 maxqsize=queue_size,
                                 checkperiod=5,
                                 verbose=1)
