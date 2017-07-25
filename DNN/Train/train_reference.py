

from training_base import training_base
from Losses import loss_NLL, loss_relMeanSquaredError, loss_NLL_mod, accuracy_SigPred,accuracy_None
from modelTools import fixLayersContaining

#also dows all the parsing
train=training_base(testrun=False)

from DeepHGCal_models import HGCal_model_reg

#if not train.modelSet():
train.setModel(HGCal_model_reg,dropoutRate=0.12)

train.compileModel(learningrate=0.001,
               loss=['categorical_crossentropy',loss_relMeanSquaredError],
               metrics=['accuracy'],
               loss_weights=[1., 1.])

train.trainedepoches=0
train.train_data.filesPreRead=6

#model,history = train.trainModel(nepochs=1, 
#                                 batchsize=500, 
#                                 stop_patience=300, 
#                                 lr_factor=0.9, 
#                                 lr_patience=2, 
#                                 lr_epsilon=0.0001, 
#                                 lr_cooldown=4, 
#                                 lr_minimum=0.0001, 
#                                 maxqsize=500)
#
#print('fixing all batch normalisation layers')
#train.keras_model = fixLayersContaining(train.keras_model,'batchnorm')
#
#train.compileModel(learningrate=0.004,
#               loss=['categorical_crossentropy',loss_relMeanSquaredError],
#               metrics=['accuracy'],
#               loss_weights=[1., 1.])
#
#
#train.trainedepoches=0

model,history = train.trainModel(nepochs=100, 
                                 batchsize=500, 
                                 stop_patience=300, 
                                 lr_factor=0.9, 
                                 lr_patience=3, 
                                 lr_epsilon=0.0001, 
                                 lr_cooldown=2, 
                                 lr_minimum=0.0001, 
                                 maxqsize=500)

