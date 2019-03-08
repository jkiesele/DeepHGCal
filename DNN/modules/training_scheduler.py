

import collections
import keras.backend as K
    
Learning_sched = collections.namedtuple('Learning_sched', 'lr nepochs batchsize funccall loss_weights en_loss')



def scheduled_training(learn, train,clipnorm,usemetrics, **trainargs):

    totalepochs=0
    trainepochs=0
    for lrs in learn:
        
        
        
        if train.trainedepoches>totalepochs:
            print('skipping already trained epochs: '+str(train.trainedepoches))
            remainingepochs=lrs.nepochs
            if lrs.nepochs+totalepochs<train.trainedepoches:
                totalepochs+=lrs.nepochs
                trainepochs=0
            else:
                trainepochs=totalepochs+lrs.nepochs-train.trainedepoches
                totalepochs=train.trainedepoches
        else:
            trainepochs=lrs.nepochs
        

    
        if trainepochs > 0:
            if lrs.funccall:
                lrs.funccall(train)
            if lrs.loss_weights:
                train.compileModel(learningrate=0.01, #anyway overwritten
                   clipnorm=clipnorm,
                   loss=['mean_absolute_error',lrs.en_loss],#'mean_squared_error'],#'mean_squared_error'],#huber_loss],#mean_squared_logarithmic_error],
                   metrics=usemetrics,
                   loss_weights=lrs.loss_weights)
                print(train.keras_model.summary())
            
            for l in train.keras_model.layers:
                if l.trainable and l.weights and len(l.weights):
                    print('trainable '+l.name)
                    
            K.set_value(train.keras_model.optimizer.lr, lrs.lr)
            print('set learning rate to '+str(lrs.lr))
            
            
                
            print('training for epochs: '+str(trainepochs))
            
            model,history = train.trainModel(nepochs=totalepochs+trainepochs, 
                                     batchsize=int(lrs.batchsize),
                                     **trainargs)
            totalepochs+=lrs.nepochs
            remainingepochs=0
    
    