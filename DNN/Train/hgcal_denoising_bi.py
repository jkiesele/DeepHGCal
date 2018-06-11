
from DeepJetCore.training.training_base import training_base

from keras.layers import Dense, ZeroPadding3D,Conv3D,Dropout, Flatten, Convolution2D,Convolution3D, merge, Convolution1D, Conv2D, LSTM, LocallyConnected2D, Bidirectional
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
from keras.layers.core import Reshape, Permute
from Layers import PermuteBatch, ReshapeBatch

l2_lambda = 0.0001


batch_size = 35
def denoising_model(Inputs, nclasses, nregressions, dropoutRate=0.05, momentum=0.6):
    global batch_size
    # image = Input(shape=(25, 25, 25, 1))

    x = Inputs[0]

    # Shape: [B,13,13,L,C]
    shape = K.int_shape(x)
    _, _, _, L, C = shape


    x = Conv3D(30, kernel_size=(1, 1, 1), padding='same', activation='relu',
                      kernel_initializer='lecun_uniform', kernel_regularizer=l2(l2_lambda))(x)
    x = Conv3D(30, kernel_size=(1, 1, 1), padding='same', activation='relu',
                      kernel_initializer='lecun_uniform', kernel_regularizer=l2(l2_lambda))(x)
    x = Conv3D(30, kernel_size=(1, 1, 1), padding='same', activation='relu',
                      kernel_initializer='lecun_uniform', kernel_regularizer=l2(l2_lambda))(x)
    x = Conv3D(30, kernel_size=(1, 1, 1), padding='same', activation='relu',
                      kernel_initializer='lecun_uniform', kernel_regularizer=l2(l2_lambda))(x)
    # Shape: [(B*L), 13, 13, 30]

    x = Conv3D(30, kernel_size=(5, 5, 1), padding='same', activation='relu',
                      kernel_initializer='lecun_uniform', kernel_regularizer=l2(l2_lambda))(x)
    x = Conv3D(30, kernel_size=(5, 5, 1), padding='same', activation='relu',
                      kernel_initializer='lecun_uniform', kernel_regularizer=l2(l2_lambda))(x)
    x = Conv3D(30, kernel_size=(5, 5, 1), padding='same', activation='relu',
                      kernel_initializer='lecun_uniform', kernel_regularizer=l2(l2_lambda))(x)
    x = Conv3D(30, kernel_size=(5, 5, 1), padding='same', activation='relu',
                      kernel_initializer='lecun_uniform', kernel_regularizer=l2(l2_lambda))(x)
    x = ReshapeBatch((-1 , L , 30))(x)

    lstm_1 = LSTM(100, return_sequences=True, input_shape=(L, 30))
    lstm_2 = Bidirectional(LSTM(80, return_sequences=True, input_shape=(L, 100)))

    x = lstm_1(x)
    x = lstm_2(x)
    x = ReshapeBatch((-1, 13, 13, L, 160))(x)
    x = Conv3D(1, kernel_size=(1, 1, 1), padding='same', activation=None,
                      kernel_initializer='lecun_uniform', kernel_regularizer=l2(l2_lambda))(x)

    model = Model(inputs=Inputs, outputs=x)
    return model

def masked_mean_square(truth, prediction):
    truth_data = truth[0]
    mask = truth[1]
    return K.mean(mask * K.square(truth_data - prediction), axis=-1)


inputs = []
shapes = [(13, 13, 55, 26)]
for s in shapes:
    inputs.append(keras.layers.Input(shape=s))

model = denoising_model(inputs, 5, 2)

print(model.summary())

#also dows all the parsing
train=training_base(testrun=False,resumeSilently=True)
train.setModel(denoising_model,dropoutRate=0.05,momentum=0.9)
train.compileModel(learningrate=0.0020,
               loss=[keras.losses.mean_squared_error])

print(train.keras_model.summary())
#train.train_data.maxFilesOpen=4
#exit()


model,history = train.trainModel(nepochs=100,
                                 batchsize=batch_size,
                                 stop_patience=300,
                                 lr_factor=0.3,
                                 lr_patience=-6,
                                 lr_epsilon=0.001,
                                 lr_cooldown=8,
                                 lr_minimum=0.000001,
                                 maxqsize=20,
                                 checkperiod=1,
                                 verbose=1)
