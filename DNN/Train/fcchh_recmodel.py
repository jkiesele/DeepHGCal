

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
from keras.layers import Cropping3D, LSTM, Masking
from keras.regularizers import l2
l2_lambda = 0.0001

def simpleRecurrent(Inputs,nclasses,nregressions,dropoutRate=0.05,momentum=0.6):
    
    x = Inputs[0]
    #mask the zero entries
    #x = Masking(mask_value=0.0)(x)
    #x = LSTM (64,go_backwards=False,implementation=1, name='listlstm_0')(x)
    x = Flatten()(x)
    #x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='dense_0')(x)
    x = Dropout(dropoutRate)(x)
    idpred = Dense(nclasses, activation='relu',kernel_initializer='lecun_uniform', name='IDpred')(x)
    epred = Dense(nregressions, activation='relu',kernel_initializer='lecun_uniform', name='Epred')(x)
    predictions = [idpred,epred]
    
    model = Model(inputs=Inputs, outputs=predictions)
    return model

#also dows all the parsing
train=training_base(testrun=False,resumeSilently=True)


if not train.modelSet():
    from Losses import loss_modRelMeanSquaredError

    train.setModel(simpleRecurrent,dropoutRate=0.05,momentum=0.9)
    train.compileModel(learningrate=0.0020,
                   loss=['categorical_crossentropy',loss_modRelMeanSquaredError],#mean_squared_logarithmic_error],
                   metrics=['accuracy'],
                   loss_weights=[.5, 100])

print(train.keras_model.summary())
#train.train_data.maxFilesOpen=4
#exit()
model,history = train.trainModel(nepochs=1, 
                                 batchsize=512, 
                                 stop_patience=300, 
                                 lr_factor=0.3, 
                                 lr_patience=-6, 
                                 lr_epsilon=0.001, 
                                 lr_cooldown=8, 
                                 lr_minimum=0.000001, 
                                 maxqsize=1,
                                 checkperiod=1,
                                 verbose=1)
