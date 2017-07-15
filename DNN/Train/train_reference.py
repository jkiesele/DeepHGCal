

from training_base import training_base
from Losses import loss_NLL, loss_relMeanSquaredError, loss_NLL_mod, accuracy_SigPred,accuracy_None


#also dows all the parsing
train=training_base(testrun=False)

from DeepHGCal_models import HGCal_model_reg

if not train.modelSet():
    train.setModel(HGCal_model_reg,dropoutRate=0.12)

    train.compileModel(learningrate=0.008,
                   loss=['categorical_crossentropy',loss_relMeanSquaredError],
                   metrics=['accuracy'],
                   loss_weights=[1., 1.])

train.trainedepoches=0

model,history = train.trainModel(nepochs=50, 
                                 batchsize=500, 
                                 stop_patience=300, 
                                 lr_factor=0.5, 
                                 lr_patience=6, 
                                 lr_epsilon=0.0001, 
                                 lr_cooldown=4, 
                                 lr_minimum=0.0001, 
                                 maxqsize=10)
