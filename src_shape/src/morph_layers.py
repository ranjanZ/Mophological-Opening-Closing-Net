import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from keras.engine.topology import Layer
from keras.layers import initializers, constraints
import tensorflow as tf
from keras.models import Sequential
from keras.utils import conv_utils
from keras import backend as K
import numpy as np


class Erosion2D(Layer):
    '''
    Erosion 2D Layer
    for now assuming channel last
    '''

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='valid', kernel_initializer='glorot_uniform',
     kernel_constraint=None,
                 **kwargs):
        super(Erosion2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        # for we are assuming channel last
        self.channel_axis = -1

        # self.output_dim = output_dim

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
              constraint =self.kernel_constraint)

        # Be sure to call this at the end
        super(Erosion2D, self).build(input_shape)

    def call(self, x):
        outputs = K.placeholder()
        for i in range(self.num_filters):
            # erosion2d returns image of same size as x
            # so taking min over channel_axis
            out = K.min(
                self.__erosion2d(x, self.kernel[..., i],
                                  self.strides, self.padding),
                axis=self.channel_axis, keepdims=True)
            
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])

        return outputs

    def compute_output_shape(self, input_shape):
        # if self.data_format == 'channels_last':
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=1)  # self.erosion_rate[i])
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters,)

    def __erosion2d(self, x, st_element, strides, padding,
                     rates=(1, 1, 1, 1)):
        # tf.nn.erosion2d(input, filter, strides, rates, padding, name=None)
        x = tf.nn.erosion2d(x, st_element, (1, ) + strides + (1, ),
                             rates, padding.upper())
        return x



class Dilation2D(Layer):
    '''
    Dilation 2D Layer
    for now assuming channel last
    '''
    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='valid', kernel_initializer='glorot_uniform',
     kernel_constraint=None,
                 **kwargs):
        super(Dilation2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_constraint = constraints.get(kernel_constraint)

        # for we are assuming channel last
        self.channel_axis = -1

        # self.output_dim = output_dim

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
              constraint =self.kernel_constraint)

        # Be sure to call this at the end
        super(Dilation2D, self).build(input_shape)

    def call(self, x):
        # outputs = K.placeholder()
        for i in range(self.num_filters):
            # dilation2d returns image of same size as x
            # so taking max over channel_axis
            out = K.max(
                self.__dilation2d(x, self.kernel[..., i],
                                  self.strides, self.padding),
                axis=self.channel_axis, keepdims=True)
            
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])

        return outputs

    def compute_output_shape(self, input_shape):
        # if self.data_format == 'channels_last':
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=1)  # self.dilation_rate[i])
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters,)

    def __dilation2d(self, x, st_element, strides, padding,
                     rates=(1, 1, 1, 1)):
        # tf.nn.dilation2d(input, filter, strides, rates, padding, name=None)
        x = tf.nn.dilation2d(x, st_element, (1, ) + strides + (1, ),
                             rates, padding.upper())
        return x




