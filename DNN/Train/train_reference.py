from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# some private extra plots
#from  NBatchLogger import NBatchLogger

import matplotlib
#if no X11 use below
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.layers import Input
#zero padding done before
#from keras.layers.convolutional import Cropping1D, ZeroPadding1D
from keras.optimizers import SGD

## to call it from cammand lines
import sys
import os
from argparse import ArgumentParser
import shutil


# argument parsing and bookkeeping

parser = ArgumentParser('Run the training')
parser.add_argument('inputDataCollection')
parser.add_argument('outputDir')
args = parser.parse_args()

inputData = os.path.abspath(args.inputDataCollection)
outputDir=args.outputDir
# create output dir

if os.path.isdir(outputDir):
    print('output directory must not exists yet')
    raise Exception('output directory must not exists yet')

os.mkdir(outputDir)
outputDir = os.path.abspath(outputDir)
outputDir+='/'

#copy configuration to output dir

shutil.copyfile(sys.argv[0],outputDir+sys.argv[0])
shutil.copyfile('../modules/DeepHGCal_models.py',outputDir+'DeepHGCal_models.py')


print ('start')


######################### KERAS PART ######################
# configure the in/out/split etc

testrun=False

nepochs=10
batchsize=20
startlearnrate=0.0005
from DeepJet_callbacks import DeepJet_callbacks

callbacks=DeepJet_callbacks(stop_patience=300, 
                            
                            lr_factor=0.5,
                            lr_patience=2, 
                            lr_epsilon=0.003, 
                            lr_cooldown=6, 
                            lr_minimum=0.000001, 
                            
                            outputDir=outputDir)
useweights=False
splittrainandtest=0.8
maxqsize=10 #sufficient

from DataCollection import DataCollection

traind=DataCollection()

traind.readFromFile(inputData)
traind.setBatchSize(batchsize)
traind.useweights=useweights

testd=traind.split(splittrainandtest)
shapes=traind.getInputShapes()


#from from keras.models import Sequential

from keras.layers import Input
print(shapes)
inputs = [Input(shape=shapes[0],name='globals'),
          Input(shape=shapes[1],name='heatmap')]

#model = Dense_model2(inputs,traind.getTruthShape()[0],(traind.getInputShapes()[0],))

from DeepHGCal_models import HGCal_model_reg
model = HGCal_model_reg(inputs,traind.getTruthShape()[0],shapes,0.1)
print('compiling')


from keras.optimizers import Adam
adam = Adam(lr=startlearnrate)
model.compile(loss=['categorical_crossentropy','mean_absolute_percentage_error'], 
              optimizer=adam,
              metrics=['accuracy','accuracy'],
              loss_weights=[1., 0.05])#strong focus on flavour

# This stores the history of the training to e.g. allow to plot the learning curve

testd.isTrain=False
traind.isTrain=True

print('split to '+str(traind.getNBatchesPerEpoch())+' train batches and '+str(testd.getNBatchesPerEpoch())+' test batches')

print('training')

print(traind.getUsedTruth())

traind.writeToFile(outputDir+'trainsamples.dc')
testd.writeToFile(outputDir+'valsamples.dc')

# the actual training


# the actual training
model.fit_generator(traind.generator() , verbose=1,
        steps_per_epoch=traind.getNBatchesPerEpoch(), 
        epochs=nepochs,
        callbacks=callbacks.callbacks,
        validation_data=testd.generator(),
        validation_steps=testd.getNBatchesPerEpoch(), #)#,
        max_q_size=maxqsize,
        #class_weight = classweights)#,
        )



#######this part should be generarlised!

#options to use are:

model.save(outputDir+"KERAS_model.h5")

# summarize history for loss for trainin and test sample
plt.plot(callbacks.history.history['loss'])
#print(callbacks.history.history['val_loss'],history.history['loss'])
plt.plot(callbacks.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(outputDir+'learningcurve.pdf') 
#plt.show()

plt.figure(2)
plt.plot(callbacks.history.history['acc'])
#print(callbacks.history.history['val_loss'],history.history['loss'])
plt.plot(callbacks.history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(outputDir+'accuracycurve.pdf')
