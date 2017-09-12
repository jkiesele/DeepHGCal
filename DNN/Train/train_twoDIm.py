

from training_base import training_base

from keras.layers import Dense, Dropout, Flatten, Convolution2D,Convolution3D, merge, Convolution1D, Conv2D, LSTM, LocallyConnected2D
from keras.layers.pooling import MaxPooling3D
from keras.models import Model
from keras.layers.core import Reshape, Masking, Permute
from keras.layers.pooling import MaxPooling2D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate,Add,Multiply

from Losses import loss_logcosh_noUnc
from keras.layers.noise import GaussianDropout



    


def twoDimModel(Inputs,nclasses,nregressions,dropoutRate=0.05,momentum=0.6):

    rechitimage=Inputs[1]
    globals=Inputs[0]
    
    # remove total energy for now. The input needs to be used, but the following sets it to zero
    totalrecenergy=Inputs[2]
    totalrecenergy=Dense(1,kernel_initializer='zeros',trainable=False)(totalrecenergy)
    globals=Concatenate()( [globals,totalrecenergy])
    ####
    
    
    ### define the layers
    x=Convolution2D(4,kernel_size=(1,1,1),strides=(1,1,1), activation='relu',kernel_initializer='lecun_uniform')(rechitimage)
    
    x = Flatten()(x)
    x=Concatenate()( [globals,x]) 
    
    
    x = Dense(300, activation='relu',kernel_initializer='lecun_uniform')(x)
    
    
    #create the predictions. Don't change anything here
    predictID=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    predictE=Dense(1, activation='linear',kernel_initializer='zeros',name='pred_E')(x)
    
    predictions = [predictID,predictE]
                   
    model = Model(inputs=Inputs, outputs=predictions)
    return model
    


#also dows all the parsing
train=training_base(testrun=False)


if not train.modelSet():

    train.setModel(twoDimModel,dropoutRate=0.05,momentum=0.9)

    train.compileModel(learningrate=0.0125,
                   loss=['categorical_crossentropy','mean_squared_error'],
                   metrics=['accuracy'],
                   loss_weights=[.05, 1.])

print(train.keras_model.summary())
#exit()
model,history = train.trainModel(nepochs=110, 
                                 batchsize=1000, 
                                 stop_patience=300, 
                                 lr_factor=0.5, 
                                 lr_patience=6, 
                                 lr_epsilon=0.0001, 
                                 lr_cooldown=8, 
                                 lr_minimum=0.000001, 
                                 maxqsize=100)
