## empty
import tensorflow as tf
from keras.engine.topology import Layer
import keras.backend as K
from keras.engine import InputSpec
import copy
import numpy as np
import keras
from keras.layers import Flatten
from symbol import factor
from matplotlib.pyplot import axis
#from keras.layers.merge import _Merge



class Print(Layer):
    def __init__(self, message, **kwargs):
        super(Print, self).__init__(**kwargs)
        self.message=message
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        return tf.Print(inputs,[inputs],self.message,summarize=300)
    
    def get_config(self):
        config = {'message': self.message}
        base_config = super(Print, self).get_config()
        return dict(list(base_config.items()) + list(config.items() ))


#class Simple_Sum3D(Layer):
#    def __init__(self, outshape, **kwargs):
#        super(Simple_Sum3D, outshape).__init__(**kwargs)
#        self.message=message
#    
#    def compute_output_shape(self, input_shape):
#        return input_shape
#    
#    def call(self, inputs):
#        return tf.Print(inputs,[inputs],self.message,summarize=300)
#    
#    def get_config(self):
#        config = {'message': self.message}
#        base_config = super(Simple_Sum3D, self).get_config()
#        return dict(list(base_config.items()) + list(config.items() ))
#
#


class Create_per_layer_energies(Layer):
    def __init__(self, **kwargs):
        super(Create_per_layer_energies, self).__init__(**kwargs)
        
    
    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0],input_shape[3]])
    
    def call(self, inputs):
        enonly = inputs[:,:,:,:,1]
        enonly= K.sum(enonly, axis=1, keepdims=False)
        enonly= K.sum(enonly, axis=1, keepdims=False)
        enonly *= 1./1000.
        return enonly
    
    def get_config(self):
        base_config = super(Create_per_layer_energies, self).get_config()
        return dict(list(base_config.items()))


class Sum_difference_squared(Layer):
    def __init__(self, **kwargs):
        super(Sum_difference_squared, self).__init__(**kwargs)
        
        
    def compute_output_shape(self, input_shape):
        print(input_shape)
        if int(input_shape[0][1]) % int(input_shape[1][3]):
            raise Exception("Sum_difference_squared shapes don't match: "+str(input_shape[0][3])+" "+str(int(input_shape[1][3])))
        
        if input_shape[0][0]>0:
            return tuple(input_shape[0][0]+[1])
        else:
            return tuple([None,1])
    
    def call(self, inputs):
        
        if inputs[0].get_shape()[0]>0:
            nbatch=int(inputs[0].shape[0])
        else:
            nbatch=-1
            
        nlayers=int(inputs[0].shape[1])
        perlayerenergies=inputs[0]
        print('perlayerenergies ',perlayerenergies.shape)
        pred_energies=inputs[1][:,:,:,:,0]
        pred_energies=tf.reduce_sum(pred_energies, axis=1, keep_dims=False)
        pred_energies=tf.reduce_sum(pred_energies, axis=1, keep_dims=False)
        print('pred_energies ',pred_energies.shape)
        
        ncombinedlayers=int(pred_energies.shape[1])
        strides = nlayers/ncombinedlayers
        print('ncombinedlayers',ncombinedlayers)
        print('strides',strides)
        print('nbatch',nbatch)
        
        perlayerenergies = tf.reshape(perlayerenergies, shape=[nbatch, ncombinedlayers, strides])
        print('perlayerenergies a',perlayerenergies.shape)
        perlayerenergies = tf.reduce_sum(perlayerenergies,axis=-1)
        print('perlayerenergies b',perlayerenergies.shape)
        
        
        #perlayerenergies=tf.Print(perlayerenergies,[perlayerenergies],"perlayerenergies_"+self.name)
        #pred_energies=tf.Print(pred_energies,[pred_energies],"pred_energies_"+self.name)
        
        diff = perlayerenergies - pred_energies
        
        #diff=tf.Print(diff,[diff],"diff"+self.name)
        
        diff *= diff
        
        diff=tf.reduce_sum(diff, axis=-1, keep_dims=False)
        diff=tf.expand_dims(diff, axis=1)
        return diff
    
    def get_config(self):
        base_config = super(Sum_difference_squared, self).get_config()
        return dict(list(base_config.items()))


class Reduce_sum(Layer):
    def __init__(self, **kwargs):
        super(Log_plus_one, self).__init__(**kwargs)
        
    def compute_output_shape(self, input_shape):
        outshape=[]
        for i in range(len(input_shape)-1):
            outshape.append(input_shape[i])
        return tuple(outshape)
    
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=-1, keep_dims=False)
    
    def get_config(self):
        base_config = super(Reduce_sum, self).get_config()
        return dict(list(base_config.items()))
    

class Clip(Layer):
    def __init__(self, min, max , **kwargs):
        super(Clip, self).__init__(**kwargs)
        self.min=min
        self.max=max
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        return tf.clip_by_value(inputs, self.min, self.max)
    
    def get_config(self):
        config = {'min': self.min, 'max': self.max}
        base_config = super(Clip, self).get_config()
        return dict(list(base_config.items()) + list(config.items() ))
    



class ScalarMultiply(Layer):
    def __init__(self, factor, **kwargs):
        super(ScalarMultiply, self).__init__(**kwargs)
        self.factor=factor
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        return inputs*self.factor
    
    def get_config(self):
        config = {'factor': self.factor}
        base_config = super(ScalarMultiply, self).get_config()
        return dict(list(base_config.items()) + list(config.items() ))
    
class Multiply_feature(Layer):
    def __init__(self, feat, factor, **kwargs):
        super(Multiply_feature, self).__init__(**kwargs)
        self.factor=factor
        self.feat=feat
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    #not very generic!
    def call(self, inputs):
        feat_input = inputs[:,:,:,:,self.feat] 
        feat_input = tf.expand_dims(feat_input, axis=4)
        other_input_a = inputs[:,:,:,:,0:self.feat] 
        other_input_b = inputs[:,:,:,:,self.feat+1:] 
        feat_input *= self.factor
        
        return tf.concat([other_input_a,feat_input,other_input_b],axis=-1)
    
    def get_config(self):
        config = {'feat': self.feat, 'factor': self.factor}
        base_config = super(Multiply_feature, self).get_config()
        return dict(list(base_config.items()) + list(config.items() ))

class Log_plus_one(Layer):
    def __init__(self, **kwargs):
        super(Log_plus_one, self).__init__(**kwargs)
        
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        return tf.log(inputs+1)
    
    def get_config(self):
        base_config = super(Log_plus_one, self).get_config()
        return dict(list(base_config.items()))
    

class SelectEnergyOnly(Layer):
    def __init__(self, **kwargs):
        super(SelectEnergyOnly, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        outshape=[]
        for i in range(len(input_shape)-1):
            outshape.append(input_shape[i])
        outshape.append(1)
        return tuple(outshape)
    
    def call(self, inputs):
        return K.expand_dims(inputs[:,:,:,:,1], axis = 4)
        
    
    def get_config(self):
        base_config = super(SelectEnergyOnly, self).get_config()
        return dict(list(base_config.items()))

class SelectFeatureOnly(Layer):
    def __init__(self, feat, **kwargs):
        super(SelectFeatureOnly, self).__init__(**kwargs)
        self.feat=feat
    
    def compute_output_shape(self, input_shape):
        outshape=[]
        for i in range(len(input_shape)-1):
            outshape.append(input_shape[i])
        outshape.append(1)
        return tuple(outshape)
    
    def call(self, inputs):
        return K.expand_dims(inputs[:,:,:,:,self.feat], axis = 4)
    
    def get_config(self):
        config = {'feat': self.feat}
        base_config = super(SelectFeatureOnly, self).get_config()
        return dict(list(base_config.items()) + list(config.items() ))

class Sum3DFeatureOne(Layer):
    def __init__(self,featindex=1, **kwargs):
        super(Sum3DFeatureOne, self).__init__(**kwargs)
        self.featindex = featindex
        
    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0],1])

    def call(self, inputs):
        permtolist=Flatten()(inputs[:,:,:,:,self.featindex])
        out= K.sum(permtolist, axis=-1, keepdims=True)
        return out

    def get_config(self):
        config = {'featindex': self.featindex}
        base_config = super(Sum3DFeatureOne, self).get_config()
        return dict(list(base_config.items()) + list(config.items() ))


class Sum3DFeaturePerLayer(Layer):
    def __init__(self,layerindex,featindex=1, **kwargs):
        super(Sum3DFeaturePerLayer, self).__init__(**kwargs)
        self.featindex = featindex
        self.layerindex=layerindex
        
    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0],1])

    def call(self, inputs):
        permtolist=Flatten()(inputs[:,:,:,self.layerindex,self.featindex])
        out= K.sum(permtolist, axis=-1, keepdims=True)/1000
        return out

    def get_config(self):
        config = {'layerindex': self.layerindex,'featindex': self.featindex}
        base_config = super(Sum3DFeaturePerLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items() ))




class PermuteBatch(Layer):
    """Permutes the dimensions of the input according to a given pattern.

    Useful for e.g. connecting RNNs and convnets together.

    # Example

    ```python
        model = Sequential()
        model.add(Permute((2, 1), input_shape=(10, 64)))
        # now: model.output_shape == (None, 64, 10)
        # note: `None` is the batch dimension
    ```

    # Arguments
        dims: Tuple of integers. Permutation pattern, includes the
            samples dimension. Indexing starts at 0.
            For instance, `(1, 0)` permutes the first and second dimension
            of the input.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, includes the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same as the input shape, but with the dimensions re-ordered according
        to the specified pattern.
    """

    def __init__(self, dims, **kwargs):
        super(PermuteBatch, self).__init__(**kwargs)
        self.dims = tuple(dims)
        self.input_spec = InputSpec(ndim=len(self.dims))

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        output_shape = copy.copy(input_shape)
        for i, dim in enumerate(self.dims):
            target_dim = input_shape[dim]
            output_shape[i] = target_dim
        return tuple(output_shape)

    def call(self, inputs):
        return K.permute_dimensions(inputs, self.dims)

    def get_config(self):
        config = {'dims': self.dims}
        base_config = super(PermuteBatch, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




class ReshapeBatch(Layer):
    """Reshapes an output to a certain shape.

    # Arguments
        target_shape: target shape. Tuple of integers.
            Includes the batch axis.

    # Input shape
        Arbitrary, although all dimensions in the input shaped must be fixed.
        Use the keyword argument `input_shape`
        (tuple of integers, includes the batch axis)
        when using this layer as the first layer in a model.

    # Output shape
        `target_shape`

    """

    def __init__(self, target_shape, **kwargs):
        super(ReshapeBatch, self).__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def compute_output_shape(self, input_shape):
        return self.target_shape

    def call(self, inputs):
        if not self.target_shape[0]:
            return K.reshape(inputs, tuple([-1]+list(self.target_shape[1:])))
        else:
            return K.reshape(inputs, self.target_shape)

    def get_config(self):
        config = {'target_shape': self.target_shape}
        base_config = super(ReshapeBatch, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

global_layers_list = {}
global_layers_list['Create_per_layer_energies']=Create_per_layer_energies
global_layers_list['Sum_difference_squared']=Sum_difference_squared
global_layers_list['Reduce_sum']=Reduce_sum
global_layers_list['SelectFeatureOnly']=SelectFeatureOnly
global_layers_list['Multiply_feature']=Multiply_feature
global_layers_list['Print']=Print
global_layers_list['Clip'] = Clip
global_layers_list['ScalarMultiply'] = ScalarMultiply
global_layers_list['Log_plus_one'] = Log_plus_one
global_layers_list['SelectEnergyOnly'] = SelectEnergyOnly
global_layers_list['Sum3DFeatureOne'] = Sum3DFeatureOne
global_layers_list['Sum3DFeaturePerLayer'] = Sum3DFeaturePerLayer
global_layers_list['ReshapeBatch'] = ReshapeBatch
global_layers_list['PermuteBatch'] = PermuteBatch

