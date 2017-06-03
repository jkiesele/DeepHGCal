
from keras.layers import Dense, Dropout, Flatten, Convolution2D, merge, Convolution1D, Conv2D, LSTM, LocallyConnected2D
from keras.models import Model
from keras.layers.core import Reshape, Masking, Permute
from keras.layers.pooling import MaxPooling2D



def HGCal_model_reg(Inputs,nclasses,Inputshape,dropoutRate=0.25):
    """
    very simple to test with
    """
    
    x=Flatten()(Inputs[1])
    globals=Inputs[0]
    
    x=merge( [globals,x] , mode='concat')
    
    #  Here add e.g. the normal dense stuff from DeepCSV
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    #x = Dropout(dropoutRate)(x)
    #x =  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    #x = Dropout(dropoutRate)(x)
    #x =  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    
    
    
    predictID=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    predictE=Dense(1, activation='relu',kernel_initializer='normal',name='E_pred')(x)
   
    predictions = [predictID,predictE]
                   
    model = Model(inputs=Inputs, outputs=predictions)
    return model
