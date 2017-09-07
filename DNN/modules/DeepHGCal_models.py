from keras.layers import Dense, Dropout, Flatten, Convolution2D, Convolution3D, merge, Convolution1D, Conv2D, LSTM, \
    LocallyConnected2D
from keras.layers.pooling import MaxPooling3D
from keras.models import Model
from keras.layers.core import Reshape, Masking, Permute
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate, Add


def HGCal_model_reg(Inputs, nclasses, Inputshape, dropoutRate=0.25):
    """
    very simple to test with
    """

    x = Inputs[1]
    globals = Inputs[0]
    x = BatchNormalization(momentum=0.3, name='input_batchnorm', center=False)(x)
    globals = BatchNormalization(momentum=0.3, name='globalinput_batchnorm')(globals)

    x = Convolution3D(24, kernel_size=(3, 3, 8), strides=(1, 1, 2), padding='same',
                      activation='relu', kernel_initializer='lecun_uniform', use_bias=False, name='conv3D_0')(x)
    x = BatchNormalization(momentum=0.3, name='conv_batchnorm0')(x)
    x = Dropout(dropoutRate)(x)
    x = Convolution3D(16, kernel_size=(7, 7, 7), strides=(4, 4, 4), padding='same', use_bias=False,
                      activation='relu', kernel_initializer='lecun_uniform', name='conv3D_1')(x)
    x = BatchNormalization(momentum=0.3, name='conv_batchnorm1')(x)
    x = Dropout(dropoutRate)(x)
    x = Convolution3D(8, kernel_size=(3, 3, 3), padding='same',
                      activation='relu', kernel_initializer='lecun_uniform', use_bias=False, name='conv3D_2')(x)
    x = BatchNormalization(momentum=0.3, name='conv_batchnorm2')(x)
    x = Dropout(dropoutRate)(x)
    x = Convolution3D(4, kernel_size=(1, 1, 1), padding='same',
                      activation='relu', kernel_initializer='lecun_uniform', use_bias=False, name='conv3D_3')(x)
    x = BatchNormalization(momentum=0.3, name='conv_batchnorm3')(x)
    x = Dropout(dropoutRate)(x)

    doreg = False
    if doreg:
        recpart = Convolution3D(1, kernel_size=(1, 1, 1), activation='relu', kernel_initializer='lecun_uniform')(x)
        recpart = Dropout(dropoutRate)(recpart)

        recpart = Reshape((-1, 7))(recpart)
        recpart = Permute((2, 1))(recpart)
        recpart = LSTM(64, return_sequences=False, implementation=2,
                       dropout=dropoutRate, recurrent_dropout=dropoutRate)(recpart)

        globals = Concatenate()([globals, recpart])

    x = Flatten()(x)
    # longrange=Convolution3D(8,kernel_size=(8,8,11), activation='relu',kernel_initializer='lecun_uniform')(Inputs[1])
    # longrange = Dropout(dropoutRate)(longrange)
    # longrange=Convolution3D(2,kernel_size=(1,1,1), activation='relu',kernel_initializer='lecun_uniform')(Inputs[1])
    # longrange = Dropout(dropoutRate)(longrange)
    # longrange = MaxPooling3D(pool_size=(5, 5, 15))(longrange)


    # longrange=Flatten()(longrange)



    # x=Convolution3D(8,kernel_size=(8,8,8), activation='relu',kernel_initializer='lecun_uniform')(Inputs[1])
    # x = Dropout(dropoutRate)(x)
    # x=Convolution3D(2,kernel_size=(8,8,8),strides=(3,3,3), activation='relu',kernel_initializer='lecun_uniform')(Inputs[1])
    # x = Dropout(dropoutRate)(x)
    # id=Flatten()(x)

    # x =  Reshape(target_shape=(64,-1))(x)
    # x  = LSTM(100,go_backwards=True)(x)

    merged = Concatenate()([globals, x])

    x = Dense(128, activation='relu', kernel_initializer='lecun_uniform')(merged)
    x = BatchNormalization(momentum=0.3, name='dense_batchnorm0')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(128, activation='relu', kernel_initializer='lecun_uniform')(x)
    # x=BatchNormalization(momentum=0.3,name='dense_batchnorm1')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(64, activation='relu', kernel_initializer='lecun_uniform')(x)
    x = BatchNormalization(momentum=0.3, name='dense_batchnorm2')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(64, activation='relu', kernel_initializer='lecun_uniform')(x)
    # x=BatchNormalization(momentum=0.3,name='dense_batchnorm3')(x)
    # x = Dropout(dropoutRate)(x)



    predictID = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', name='ID_pred')(x)
    predictE = Dense(1, activation='linear', kernel_initializer='zeros', name='E_pred_E')(x)
    predictS = Dense(1, activation='linear', kernel_initializer='ones', name='E_pred_S')(x)
    EandS = Concatenate(name='E_pred')([predictS, predictE])

    predictions = [predictID, EandS]

    model = Model(inputs=Inputs, outputs=predictions)
    return model
