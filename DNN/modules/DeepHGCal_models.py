
from keras.layers import Dense, Dropout, Flatten, Convolution2D,Convolution3D, merge, Convolution1D, Conv2D, LSTM, LocallyConnected2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.layers.core import Reshape, Masking, Permute
from keras.layers.pooling import MaxPooling2D



def HGCal_model_reg(Inputs,nclasses,Inputshape,dropoutRate=0.25):
    """
    very simple to test with
    """
    
    x=Inputs[1]
    
    #first shower z direction
    x=Convolution2D(64,kernel_size=(3,3), activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=Convolution2D(64,kernel_size=(3,3), activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=Convolution2D(64,kernel_size=(6,6), activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=Convolution2D(5,kernel_size=(6,6),strides=(3,3), activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)

    x = Flatten()(x)
    #x =  Reshape(target_shape=(64,-1))(x)
    #x  = LSTM(100,go_backwards=True)(x)
    globals=Inputs[0]
    
    x=merge( [globals,x] , mode='concat')
    
    #  Here add e.g. the normal dense stuff from DeepCSV
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    
    
    
    predictID=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    predictE=Dense(2, activation='linear',kernel_initializer='ones',name='E_pred')(x)
   
    predictions = [predictID,predictE]
                   
    model = Model(inputs=Inputs, outputs=predictions)
    return model
