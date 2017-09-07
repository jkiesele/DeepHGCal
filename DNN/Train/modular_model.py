

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


def submodule_model(Inputs,nclasses,nregressions,dropoutRate=0.05,momentum=0.6):
    
    x=Inputs[1]
    globals=Inputs[0]
    totalrecenergy=Inputs[2]
    totalrecenergy= GaussianDropout(0.03)(totalrecenergy)
    
    ########## transform inputs per pixel (relatively cheap - low risk of overtraining) #########
    x=Convolution3D(32,kernel_size=(1,1,1),strides=(1,1,1),
                    padding='same', activation='relu',kernel_initializer='lecun_uniform',name='conv_pre0')(x)
    x=Dropout(dropoutRate)(x)                                                      
    x=Convolution3D(32,kernel_size=(1,1,1),strides=(1,1,1),                            
                    padding='same', activation='relu',kernel_initializer='lecun_uniform',name='conv_pre1')(x)
    x=Dropout(0.5*dropoutRate)(x)                                                     
    x=Convolution3D(32,kernel_size=(1,1,1),strides=(1,1,1),                            
                    padding='same', activation='relu',kernel_initializer='lecun_uniform',name='conv_pre2')(x)
    x=Dropout(0.5*dropoutRate)(x)                                                      
    x=Convolution3D(12,kernel_size=(1,1,1),strides=(1,1,1),                              
                    padding='same', activation='relu',kernel_initializer='lecun_uniform',name='conv_pre3')(x)
    x=BatchNormalization(momentum=momentum,name='bn_postPre')(x)                                                           
                                                                                        
    ######### make a 2D search for features ############                                
    x=Convolution3D(32,kernel_size=(5,5,1),strides=(1,1,1),                             
                    padding='same', activation='relu',kernel_initializer='lecun_uniform',name='conv_2D0')(x)
    x=Dropout(dropoutRate)(x)
    x=Convolution3D(52,kernel_size=(5,5,1),strides=(3,3,1),                             
                    padding='same', activation='relu',kernel_initializer='lecun_uniform',name='conv_2D1')(x)
    x=Dropout(dropoutRate)(x)                                
    x=Convolution3D(32,kernel_size=(1,1,1),strides=(1,1,1),                             
                    padding='same', activation='relu',kernel_initializer='lecun_uniform',name='conv_2D2')(x)
     
    x=BatchNormalization(momentum=momentum,name='bn_post2D')(x)
    x=Dropout(dropoutRate)(x) 
                                                                                        
    ########## make layer search                                                        
    x=Convolution3D(64,kernel_size=(1,1,5),strides=(1,1,1),                             
                    padding='same', activation='relu',kernel_initializer='lecun_uniform',name='conv_Z0')(x)
    x=Dropout(dropoutRate)(x)
    x=Convolution3D(52,kernel_size=(1,1,9),strides=(1,1,4),                             
                    padding='same', activation='relu',kernel_initializer='lecun_uniform',name='conv_Z1')(x)
    x=Dropout(dropoutRate)(x)
    x=Convolution3D(52,kernel_size=(1,1,9),strides=(1,1,3),                             
                    padding='same', activation='relu',kernel_initializer='lecun_uniform',name='conv_Z2')(x)
    x=Dropout(dropoutRate)(x)
    x=Convolution3D(32,kernel_size=(1,1,1),strides=(1,1,1),                            
                    padding='same', activation='relu',kernel_initializer='lecun_uniform',name='conv_Z3')(x)
                    
    x=BatchNormalization(momentum=momentum,name='bn_postZ')(x)
    x=Dropout(dropoutRate)(x) 
    ########## bring everything together one round ##########
    x=Convolution3D(32,kernel_size=(3,3,3),strides=(2,2,2),                             
                    padding='same', activation='relu',kernel_initializer='lecun_uniform',name='conv_3D0')(x)
    x=Dropout(dropoutRate)(x)
    x=Convolution3D(32,kernel_size=(1,1,1),strides=(1,1,1),                             
                    padding='same', activation='relu',kernel_initializer='lecun_uniform',name='conv_3D1')(x)
    x=Dropout(0.5*dropoutRate)(x)
    x=Convolution3D(12,kernel_size=(1,1,1),strides=(1,1,1),                             
                    padding='same', activation='relu',kernel_initializer='lecun_uniform',name='conv_3D2')(x)
    
    x=BatchNormalization(momentum=momentum,name='bn_post3D')(x) 
    
    
    x=Flatten()(x)
    x=Dropout(dropoutRate)(x)
    x=Concatenate()([x,globals])
    x=Dense(512, activation='relu',kernel_initializer='lecun_uniform',name='dense_0')(x)
    x=Dropout(dropoutRate)(x)

    predictID=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    predictE=Dense(1, activation='linear',kernel_initializer='zeros',name='pred_E_corr')(x)
    predictE = Add(name='addingE')([totalrecenergy,predictE])
    predictE=Dense(1, activation='linear',kernel_initializer='ones',name='pred_E')(predictE)
    
    predictions = [predictID,predictE]
                   
    model = Model(inputs=Inputs, outputs=predictions)
    return model

#also dows all the parsing
train=training_base(testrun=False)

train.train_data.maxFilesOpen=4

if not train.modelSet():

    train.setModel(submodule_model,dropoutRate=0.06,momentum=0.9)

    train.compileModel(learningrate=0.02,
                   loss=['categorical_crossentropy','mean_squared_error'],
                   metrics=['accuracy'],
                   loss_weights=[.1, 1.])

print(train.keras_model.summary())


model,history = train.trainModel(nepochs=100, 
                                 batchsize=192, 
                                 stop_patience=300, 
                                 lr_factor=0.7, 
                                 lr_patience=-6, 
                                 lr_epsilon=0.0001, 
                                 lr_cooldown=10, 
                                 lr_minimum=0.000001, 
                                 maxqsize=200)
