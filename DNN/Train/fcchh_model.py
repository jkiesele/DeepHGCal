

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
l2_lambda = 0.001


############# just a placeholder

def sofiamodel(Inputs,nclasses,nregressions,dropoutRate=0.05,momentum=0.6):
    #image = Input(shape=(25, 25, 25, 1))
    
    regulariser=l2(l2_lambda)
 
    #34x34x8x2
    ecalhits= BatchNormalization(momentum=momentum)(Inputs[0])
    ecalhits = Dropout(dropoutRate)(ecalhits)
    
    #17x17x10x2
    hcalhits= BatchNormalization(momentum=momentum)(Inputs[1])
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
    fullcaloimage = BatchNormalization()(fullcaloimage)
    fullcaloimage = Dropout(dropoutRate)(fullcaloimage)
    
    #non-muon ID part
    
    x = Convolution3D(16,kernel_size=(5, 5, 5),strides=(1,1,1), activation='relu',name='conv1',
                      kernel_initializer='lecun_uniform',kernel_regularizer=regulariser,
                      border_mode='same')(fullcaloimage)
    x = BatchNormalization()(x)
    x = Dropout(dropoutRate)(x)
    
    x = Convolution3D(12,kernel_size=(5, 5, 5),strides=(1,1,1), activation='relu',name='conv2',
                      kernel_initializer='lecun_uniform',
                      border_mode='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropoutRate)(x)
    
    x = Convolution3D(8,kernel_size=(5, 5, 5),strides=(2,2,2), activation='relu',name='conv3',
                      kernel_initializer='lecun_uniform',
                      border_mode='valid')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropoutRate)(x)
    
    x = Convolution3D(16,kernel_size=(5, 5, 5),strides=(2,2,2), activation='relu',name='conv4',
                      kernel_initializer='lecun_uniform',
                      border_mode='valid')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropoutRate)(x)
    
    #x = AveragePooling3D((2, 2, 6),name="avpool")(x)
    x = Flatten()(x)
    
    if False:
    # muon part
        smallshower = Cropping3D(cropping=((4,4),(4,4),(0,0)),name='muoncrop')(fullcaloimage)
        
        smallshower=Convolution3D(32, (3, 3, 5),strides=(2,2,3) , border_mode='same',activation='relu',
                                  kernel_regularizer=regulariser,name='muon0' )(smallshower)
        smallshower = Dropout(dropoutRate)(smallshower)          
                        
        smallshower=Convolution3D(16 , (1, 1, 5),strides=(1,1,3) , border_mode='same',activation='relu',
                                  name='muon1' )(smallshower)
        smallshower = BatchNormalization()(smallshower)
        smallshower = Dropout(dropoutRate)(smallshower)
        
        smallshower=Convolution3D(4  , (1, 1, 5),strides=(1,1,3) , border_mode='same',activation='relu',
                                  name='muon2' )(smallshower)
        smallshower = BatchNormalization()(smallshower)
        smallshower = Dropout(dropoutRate)(smallshower)
        
        flattenedsmall  = Flatten()(smallshower)
        
        
        ##### bring both together
        x=Concatenate()( [x,flattenedsmall]) #add the inputs again in case some don't like the multiplications
    
    x = Dense(64, activation='relu',kernel_initializer='lecun_uniform',name='firstDense')(x)
    x = Dense(32, activation='relu',kernel_initializer='lecun_uniform',name='dense2')(x)
    x = Dense(1, activation='linear',kernel_initializer='lecun_uniform',use_bias=False)(x)   
    
    predictE=Dense(1, activation='linear',kernel_initializer='lecun_uniform',name='pred_E')(x)
    predictID=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    predictions = [predictID,predictE]
    
    model = Model(inputs=Inputs, outputs=predictions)
    return model


#also dows all the parsing
train=training_base(testrun=False,resumeSilently=True)


if not train.modelSet():
    from Losses import loss_modRelMeanSquaredError

    train.setModel(sofiamodel,dropoutRate=0.05,momentum=0.9)
    train.compileModel(learningrate=0.001,
                   loss=['categorical_crossentropy',loss_modRelMeanSquaredError],#mean_squared_logarithmic_error],
                   metrics=['accuracy'],
                   loss_weights=[0., 0.01])

print(train.keras_model.summary())


#train.train_data.maxFilesOpen=4
#exit()
model,history = train.trainModel(nepochs=60, 
                                 batchsize=1800, 
                                 stop_patience=300, 
                                 lr_factor=0.3, 
                                 lr_patience=-6, 
                                 lr_epsilon=0.001, 
                                 lr_cooldown=8, 
                                 lr_minimum=0.000001, 
                                 maxqsize=7,
                                 checkperiod=5,
                                 verbose=1)
