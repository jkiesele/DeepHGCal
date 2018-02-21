
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
import keras
from keras import backend as K

l2_lambda = 0.0001


def denoising_model(Inputs, nclasses, nregressions, dropoutRate=0.05, momentum=0.6):
    # image = Input(shape=(25, 25, 25, 1))

    x = Inputs[0]

    x = Conv3D(32, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu',
                      kernel_initializer='lecun_uniform', kernel_regularizer=l2(l2_lambda))(x)
    x = Dropout(dropoutRate)(x)
    x = Conv3D(12, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu',
                      kernel_initializer='lecun_uniform')(x)
    x = BatchNormalization(momentum=momentum)(x)
    preprocessed = Dropout(dropoutRate)(x)


    x = Conv3D(32, (5, 5, 5), padding='same', kernel_regularizer=l2(l2_lambda))(preprocessed)
    x = LeakyReLU()(x)
    x = Dropout(dropoutRate)(x)


    x = Conv3D(8, (5, 5, 5), padding='same', kernel_regularizer=l2(l2_lambda))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropoutRate)(x)



    x = Conv3D(8, (5, 5, 5), padding='same', kernel_regularizer=l2(l2_lambda))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropoutRate)(x)


    x = Conv3D(8, (5, 5, 5), padding='same', kernel_regularizer=l2(l2_lambda), )(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropoutRate)(x)

    x = Conv3D(1, (1, 1, 1), padding='same', activation='relu', kernel_regularizer=l2(l2_lambda), )(x)
    x = Conv3D(1, (1, 1, 1), padding='same', kernel_regularizer=l2(l2_lambda), )(x)

    model = Model(inputs=Inputs, outputs=x)
    return model

def masked_mean_square(truth, prediction):
    truth_data = truth[0]
    mask = truth[1]
    return K.mean(mask * K.square(truth_data - prediction), axis=-1)


inputs = []
shapes = [(13, 13, 52, 26)]
for s in shapes:
    inputs.append(keras.layers.Input(shape=s))

model = denoising_model(inputs, 5, 2)



#also dows all the parsing
train=training_base(testrun=False,resumeSilently=True)
train.setModel(denoising_model,dropoutRate=0.05,momentum=0.9)
train.compileModel(learningrate=0.0020,
               loss=[keras.losses.mean_squared_error])

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
