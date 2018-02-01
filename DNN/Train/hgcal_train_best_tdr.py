

from DeepJetCore.training.training_base import training_base

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
l2_lambda = 0.0001

def sofiamodel(Inputs,nclasses,nregressions,dropoutRate=0.05,momentum=0.6):
    #image = Input(shape=(25, 25, 25, 1))
 
    x=Inputs[1]
    globals=Inputs[0]
    totalrecenergy=Inputs[2]
    totalrecenergy=Dense(1,kernel_initializer='zeros',trainable=False)(totalrecenergy)
   
    
   
    x = Convolution3D(32,kernel_size=(1,1,1),strides=(1,1,1), activation='relu',kernel_initializer='lecun_uniform',kernel_regularizer=l2(l2_lambda))(x)
    x = Dropout(dropoutRate)(x)
    x = Convolution3D(12,kernel_size=(1,1,1),strides=(1,1,1), activation='relu',kernel_initializer='lecun_uniform')(x)
    x = BatchNormalization(momentum=momentum)(x)
    preprocessed = Dropout(dropoutRate)(x)

    x = Convolution3D(32, (5, 5, 5), border_mode='same',kernel_regularizer=l2(l2_lambda))(preprocessed)
    x = LeakyReLU()(x)
    x = Dropout(dropoutRate)(x)

    x = ZeroPadding3D((2, 2,2))(x)
    x = Convolution3D(8, (5, 5, 5), border_mode='valid',kernel_regularizer=l2(l2_lambda))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropoutRate)(x)

    x = ZeroPadding3D((2, 2, 2))(x)
    x = Convolution3D(8, (5, 5,5), border_mode='valid',kernel_regularizer=l2(l2_lambda))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropoutRate)(x)

    x = ZeroPadding3D((1, 1, 1))(x)
    x = Convolution3D(8, (5, 5, 5), border_mode='valid',kernel_regularizer=l2(l2_lambda), )(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropoutRate)(x)

    x = AveragePooling3D((2, 2, 6))(x)
    x = Flatten()(x)
    
    smallshower = Cropping3D(cropping=((4,4),(4,4),(0,0)),name='muoncrop')(preprocessed)
    smallshower=Convolution3D(32, (3, 3, 5),strides=(2,2,3) , border_mode='same',name='muon0',kernel_regularizer=l2(l2_lambda), )(smallshower)
    smallshower = LeakyReLU()(smallshower)
    smallshower = BatchNormalization()(smallshower)
    smallshower = Dropout(dropoutRate)(smallshower)
    smallshower=Convolution3D(16 , (1, 1, 5),strides=(1,1,3) , border_mode='same',name='muon1',kernel_regularizer=l2(l2_lambda), )(smallshower)
    smallshower = LeakyReLU()(smallshower)
    smallshower = BatchNormalization()(smallshower)
    smallshower = Dropout(dropoutRate)(smallshower)
    smallshower=Convolution3D(4  , (1, 1, 5),strides=(1,1,3) , border_mode='same',name='muon2',kernel_regularizer=l2(l2_lambda), )(smallshower)
    smallshower = LeakyReLU()(smallshower)
    smallshower = BatchNormalization()(smallshower)
    smallshower = Dropout(dropoutRate)(smallshower)
    flattenedsmall  = Flatten()(smallshower)
    
    
    merged=Concatenate()( [globals,x,totalrecenergy,flattenedsmall]) #add the inputs again in case some don't like the multiplications
    
 
    x = Dense(32, activation='relu',kernel_initializer='lecun_uniform',name='firstDense')(merged)
    x=Dense(1, activation='linear',kernel_initializer='lecun_uniform',use_bias=False)(x)   
    predictE=Dense(1, activation='linear',kernel_initializer='lecun_uniform',name='pred_E_corr')(x)
    
    predictID=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
     
    predictions = [predictID,predictE]
    
    model = Model(inputs=Inputs, outputs=predictions)
    return model

#also dows all the parsing
train=training_base(testrun=False,resumeSilently=True)


if not train.modelSet():
    from Losses import loss_modRelMeanSquaredError

    train.setModel(sofiamodel,dropoutRate=0.05,momentum=0.9)
    train.compileModel(learningrate=0.0020,
                   loss=['categorical_crossentropy',loss_modRelMeanSquaredError],#mean_squared_logarithmic_error],
                   metrics=['accuracy'],
                   loss_weights=[.5, 100])

print(train.keras_model.summary())
#train.train_data.maxFilesOpen=4
#exit()
model,history = train.trainModel(nepochs=1, 
                                 batchsize=300, 
                                 stop_patience=300, 
                                 lr_factor=0.3, 
                                 lr_patience=-6, 
                                 lr_epsilon=0.001, 
                                 lr_cooldown=8, 
                                 lr_minimum=0.000001, 
                                 maxqsize=20,
                                 checkperiod=1,
                                 verbose=1)
