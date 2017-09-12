

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



    
    


def bestModel(Inputs,nclasses,nregressions,dropoutRate=0.05,momentum=0.6):

    x=Inputs[1]
    globals=Inputs[0]
    totalrecenergy=Inputs[2]
    
    x=Convolution3D(4,kernel_size=(1,1,1),strides=(1,1,1), activation='relu',kernel_initializer='lecun_uniform')(x)
    x=BatchNormalization(momentum=momentum)(x)
    x=Convolution3D(16,kernel_size=(3,3,3),strides=(1,1,1), padding='same', activation='relu',kernel_initializer='lecun_uniform')(x)
    x=BatchNormalization(momentum=momentum)(x)
    x = GaussianDropout(dropoutRate)(x)
    x=Convolution3D(16,kernel_size=(3,3,6),strides=(1,1,2), padding='same',activation='relu',kernel_initializer='lecun_uniform')(x)
    x=BatchNormalization(momentum=momentum)(x)
    x = GaussianDropout(dropoutRate)(x)
    x=Convolution3D(8,kernel_size=(8,8,12),strides=(2,2,4),activation='relu',kernel_initializer='lecun_uniform')(x)
    x=BatchNormalization(momentum=momentum)(x)
    x = GaussianDropout(dropoutRate)(x)
    x=Convolution3D(5,kernel_size=(1,1,1), activation='relu',kernel_initializer='lecun_uniform')(x)
    x=BatchNormalization(momentum=momentum)(x)
    x = GaussianDropout(dropoutRate)(x)
    
    totalrecenergy=Dense(1,kernel_initializer='zeros',trainable=False)(totalrecenergy)
    x = Flatten()(x)
    merged=Concatenate()( [globals,x,totalrecenergy]) #add the inputs again in case some don't like the multiplications
    
    
    x = Dense(300, activation='relu',kernel_initializer='lecun_uniform')(merged)
    x=BatchNormalization(momentum=momentum)(x)
    x = GaussianDropout(dropoutRate)(x)
    x = Dense(200, activation='relu',kernel_initializer='lecun_uniform')(merged)
    x=BatchNormalization(momentum=momentum)(x)
    x = GaussianDropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(merged)
    #x=BatchNormalization(momentum=momentum)(x)
    x = GaussianDropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(merged)
    #x=BatchNormalization(momentum=momentum)(x)
    #x = Dropout(dropoutRate)(x)
    
    predictID=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    predictE=Dense(1, activation='linear',kernel_initializer='zeros',name='pred_E')(x)
    
    predictions = [predictID,predictE]
                   
    model = Model(inputs=Inputs, outputs=predictions)
    return model
    


def complexModel(Inputs,nclasses,nregressions,dropoutRate=0.05):

    x=Inputs[1]
    globals=Inputs[0]
    x=BatchNormalization()(x)
    x=Convolution3D(16,kernel_size=(3,3,8),strides=(1,1,2),padding='same', activation='relu',kernel_initializer='lecun_uniform')(x)
    x=BatchNormalization()(x)
    x = GaussianDropout(dropoutRate)(x)
    x=Convolution3D(16,kernel_size=(9,9,9),strides=(3,3,3),padding='same', activation='relu',kernel_initializer='lecun_uniform')(x)
    x=BatchNormalization()(x)
    x = GaussianDropout(dropoutRate)(x)
    x=Convolution3D(4,kernel_size=(3,3,3),padding='same', activation='relu',kernel_initializer='lecun_uniform')(x)
    x=BatchNormalization()(x)
    x = GaussianDropout(dropoutRate)(x)
    x=Convolution3D(4,kernel_size=(1,1,1),padding='same', activation='relu',kernel_initializer='lecun_uniform')(x)
    x=BatchNormalization()(x)
    x = GaussianDropout(dropoutRate)(x)
    x = Flatten()(x)
    merged=Concatenate()( [globals,x])
    
    x = Dense(128, activation='relu',kernel_initializer='lecun_uniform')(merged)
    x=BatchNormalization()(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(128, activation='relu',kernel_initializer='lecun_uniform')(x)
    x=BatchNormalization()(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(64, activation='relu',kernel_initializer='lecun_uniform')(x)
    x=BatchNormalization()(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(64, activation='relu',kernel_initializer='lecun_uniform')(x)
    x=BatchNormalization()(x)
    x = Dropout(dropoutRate)(x)
    predictID=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    predictE=Dense(1, activation='linear',kernel_initializer='lecun_uniform',name='E_pred_E')(x)
    
    predictions = [predictID,predictE]
                   
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def maxpoolModelSimple(Inputs,nclasses,nregressions,dropoutRate=0.05,momentum=0.6):
    
    x=Inputs[1]
    globals=Inputs[0]
    x=BatchNormalization(momentum=momentum)(x)
    x = GaussianDropout(dropoutRate)(x)
    x=Convolution3D(12,kernel_size=(3,3,3),strides=(1,1,1),padding='valid', 
                    activation='relu',kernel_initializer='lecun_uniform',name='conv3D_0a')(x)
    
    
    x=Convolution3D(12,kernel_size=(3,3,3),strides=(1,1,1),padding='valid', 
                    activation='relu',kernel_initializer='lecun_uniform',name='conv3D_0')(x)
                    
    x = MaxPooling3D((2,2,2))(x)
    
    x=Convolution3D(16,kernel_size=(5,5,5),strides=(1,1,1),padding='same', 
                    activation='relu',kernel_initializer='lecun_uniform',name='conv3D_1')(x)
    
    x = MaxPooling3D((2,2,2))(x)
    x=BatchNormalization(momentum=momentum)(x)
                    
    x=Convolution3D(4,kernel_size=(3,3,5),strides=(1,1,1),padding='same', 
                    activation='relu',kernel_initializer='lecun_uniform',name='conv3D_2')(x)
                    
    x = MaxPooling3D((1,1,3))(x)
    
    x=BatchNormalization(momentum=momentum)(x)
    x=Flatten()(x)
    x=Concatenate()( [globals,x])
    x=Dense(128, activation='relu',kernel_initializer='lecun_uniform')(x)
    
    predictID=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    predictE=Dense(1, activation='linear',kernel_initializer='lecun_uniform',name='pre_Epred')(x)
    predictions = [predictID,predictE]
                   
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def mediumDumbModel(Inputs,nclasses,nregressions,dropoutRate=0.05,momentum=0.6):

    x=Inputs[1]
    globals=Inputs[0]

    x = GaussianDropout(dropoutRate)(x)
    x=Convolution3D(12,kernel_size=(3,3,3),strides=(3,3,3),padding='same', 
                    activation='relu',kernel_initializer='lecun_uniform',name='conv3D_0a')(x)
    
    x=Convolution3D(12,kernel_size=(3,3,5),strides=(3,3,5),padding='same', 
                    activation='relu',kernel_initializer='lecun_uniform',name='conv3D_0')(x)
    x=Convolution3D(4,kernel_size=(1,1,1),strides=(1,1,1),padding='same', 
                    activation='relu',kernel_initializer='lecun_uniform',name='conv3D_1')(x)
    x=Flatten()(x)
    x=Concatenate()( [globals,x])
    x=Dense(128, activation='relu',kernel_initializer='lecun_uniform')(x)
    
    predictID=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    predictE=Dense(1, activation='linear',kernel_initializer='lecun_uniform',name='pre_Epred')(x)
    predictions = [predictID,predictE]
                   
    model = Model(inputs=Inputs, outputs=predictions)
    return model
    
    
    


def dumbestModelEver(Inputs,nclasses,nregressions,dropoutRate=0.05,momentum=0.6):
    
    x=Inputs[1]
    globals=Inputs[0]
    x=Flatten()(x)
    x=Concatenate()([globals,x])
    x = GaussianDropout(dropoutRate)(x)

    predictE=Dense(1, activation='linear',kernel_initializer='ones',name='pre_Epred')(x)
    predictID=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(predictE)
    predictions = [predictID,predictE]
                   
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def sumModelSimple(Inputs,nclasses,nregressions,dropoutRate=0.05,momentum=0.6):
    
    x=Inputs[1]
    globals=Inputs[0]
    x=BatchNormalization(momentum=momentum)(x)
    x = GaussianDropout(dropoutRate)(x)
    x=Convolution3D(12,kernel_size=(3,3,3),strides=(1,1,1),padding='valid', 
                    activation='relu',kernel_initializer='lecun_uniform',name='conv3D_0a')(x)
    
    x=Convolution3D(12,kernel_size=(3,3,3),strides=(2,2,2),padding='valid', 
                    activation='relu',kernel_initializer='lecun_uniform',name='conv3D_0')(x)
    
    x=Convolution3D(16,kernel_size=(5,5,5),strides=(2,2,2),padding='same', 
                    activation='relu',kernel_initializer='lecun_uniform',name='conv3D_1')(x)
    x=BatchNormalization(momentum=momentum)(x)
                    
    x=Convolution3D(4,kernel_size=(3,3,5),strides=(2,2,3),padding='same', 
                    activation='relu',kernel_initializer='lecun_uniform',name='conv3D_2')(x)
    x=BatchNormalization(momentum=momentum)(x)
    x=Flatten()(x)
    x=Concatenate()( [globals,x])
    x=Dense(128, activation='relu',kernel_initializer='lecun_uniform')(x)
    
    predictID=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    predictE=Dense(1, activation='linear',kernel_initializer='lecun_uniform',name='pre_Epred')(x)
    predictions = [predictID,predictE]
                   
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def sumModel(Inputs,nclasses,nregressions,dropoutRate=0.25,momentum=0.6):
    
    x=Inputs[1]
    globals=Inputs[0]
    x = GaussianDropout(dropoutRate)(x)
    #x=BatchNormalization(momentum=momentum,center=False)(x)
    x=Convolution3D(12,kernel_size=(3,3,3),strides=(1,1,1),padding='valid', 
                    activation='relu',kernel_initializer='lecun_uniform',name='conv3D_0a')(x)
                    
    x=Convolution3D(12,kernel_size=(3,3,3),strides=(2,2,2),padding='same', 
                    activation='relu',kernel_initializer='lecun_uniform',name='conv3D_0')(x)
    #x=BatchNormalization(momentum=momentum,center=False)(x)
    
    x=Convolution3D(8,kernel_size=(3,3,3),strides=(2,2,2),padding='same', 
                    activation='relu',kernel_initializer='lecun_uniform',name='conv3D_1')(x)
    #x=BatchNormalization(momentum=momentum,center=False)(x)

    x=Convolution3D(8,kernel_size=(3,3,5),strides=(2,2,3),padding='same', 
                    activation='relu',kernel_initializer='lecun_uniform',name='conv3D_2')(x)
    x = GaussianDropout(dropoutRate)(x)
    
    x=Flatten()(x)
    #x=BatchNormalization(momentum=momentum,center=False)(x)
    
    x=Concatenate()( [globals,x])
    x=Dense(128, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = GaussianDropout(dropoutRate)(x)
    x=Dense(128, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = GaussianDropout(dropoutRate)(x)
    x=Dense(128, activation='relu',kernel_initializer='lecun_uniform')(x)
    
    predictID=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    predictE=Dense(1, activation='linear',kernel_initializer='lecun_uniform',name='pre_Epred')(x)
    
    predictions = [predictID,predictE]
                   
    model = Model(inputs=Inputs, outputs=predictions)
    return model
    

def HGCalModel(Inputs,nclasses,nregressions,dropoutRate=0.25,momentum=0.6):
    """
    very simple to test with
    """
    
    
    
    x=Inputs[1]
    globals=Inputs[0]
    #x=BatchNormalization(momentum=momentum,name='input_batchnorm',center=False)(x)
    #globals=BatchNormalization(momentum=momentum,name='globalinput_batchnorm')(globals)
    
    
    x=Convolution3D(32,kernel_size=(3,3,6),strides=(1,1,3),padding='same', 
                    activation='relu',kernel_initializer='lecun_uniform',name='conv3D_0')(x)
    #x=BatchNormalization(momentum=momentum,name='conv_batchnorm0',center=False)(x)
    #x = Dropout(dropoutRate)(x)
    x=Convolution3D(32,kernel_size=(6,6,6),strides=(3,3,3),padding='same', 
                    activation='relu',kernel_initializer='lecun_uniform',name='conv3D_1')(x)
    x=Convolution3D(32,kernel_size=(1,1,1),strides=(1,1,1),padding='same', 
                    activation='relu',kernel_initializer='lecun_uniform',name='conv3D_1a')(x)
    x=Convolution3D(32,kernel_size=(1,1,1),strides=(1,1,1),padding='same', 
                    activation='relu',kernel_initializer='lecun_uniform',name='conv3D_1b')(x)
    x=Convolution3D(32,kernel_size=(1,1,1),strides=(1,1,1),padding='same', 
                    activation='relu',kernel_initializer='lecun_uniform',name='conv3D_1c')(x)
    #x=BatchNormalization(momentum=momentum,name='conv_batchnorm1')(x)
    #x = Dropout(dropoutRate)(x)
    #x=Convolution3D(12,kernel_size=(4,4,4),strides=(1,1,2),padding='same', 
    #                activation='relu',kernel_initializer='lecun_uniform',name='conv3D_2')(x)
    #x=BatchNormalization(momentum=momentum,name='conv_batchnorm2')(x)
    #x = Dropout(dropoutRate)(x)
    x=Convolution3D(nclasses+nregressions,kernel_size=(1,1,1),padding='same', 
                    activation='relu',kernel_initializer='lecun_uniform',name='conv3D_3')(x)
    #x=BatchNormalization(momentum=momentum,name='conv_batchnorm3')(x)
    #x = Dropout(dropoutRate)(x)
    
    
    x = Flatten()(x)
    
    merged=Concatenate()( [globals,x])
    #merged = BatchNormalization(momentum=momentum,name='merge_batchnorm0')(merged)
    #ID part
    merged = Dense(64, activation='softplus',kernel_initializer='lecun_uniform')(merged)
    #x = BatchNormalization(momentum=momentum,name='dense_batchnorm0')(x)
    #x = Dropout(dropoutRate)(x)
    x = Dense(64, activation='relu',kernel_initializer='lecun_uniform')(x)

    predictID=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
   
   
    #energy part - more non-linear
    E = Concatenate()( [merged,predictID])
    E = Dense(64, activation='softplus',kernel_initializer='lecun_uniform')(E)
    E = Dense(64, activation='relu',kernel_initializer='lecun_uniform')(E)
    E = Dense(64, activation='softplus',kernel_initializer='lecun_uniform')(E)
    #Emulti=Dense(64, activation='tanh',kernel_initializer='ones')(E)
    #E=Multiply()([E,Emulti])
    
    
    predictE=Dense(1, activation='linear',kernel_initializer='lecun_uniform',name='E_pred_E')(E)
    predictS=Dense(1, activation='linear',kernel_initializer='ones',name='E_pred_S')(E)
    SandE=Concatenate(name='E_pred')( [predictS,predictE] )
   
    predictions = [predictID,SandE]
                   
    model = Model(inputs=Inputs, outputs=predictions)
    return model


from DeepHGCal_models import HGCal_model_reg


#also dows all the parsing
train=training_base(testrun=False)



if not train.modelSet():

    train.setModel(bestModel,dropoutRate=0.05,momentum=0.9)

    train.compileModel(learningrate=0.0125,
                   loss=['categorical_crossentropy','mean_squared_error'],
                   metrics=['accuracy'],
                   loss_weights=[.05, 1.])

print(train.keras_model.summary())
#exit()
model,history = train.trainModel(nepochs=110, 
                                 batchsize=665, 
                                 stop_patience=300, 
                                 lr_factor=0.5, 
                                 lr_patience=-6, 
                                 lr_epsilon=0.0001, 
                                 lr_cooldown=8, 
                                 lr_minimum=0.000001, 
                                 maxqsize=200)
