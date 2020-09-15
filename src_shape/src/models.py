# cod#ing=utf-8
from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Add, Dropout, concatenate
from morph_layers import *
from keras.layers import Activation, Dense

def unet_down_1(filter_count, inputs, activation='relu', pool=(2, 2), n_layers=3):
    down = inputs
    for i in range(n_layers):
        down = Conv2D(filter_count, (3, 3), padding='same', activation=activation)(down)
        down = BatchNormalization()(down)

    if pool is not None:
        x = MaxPooling2D(pool, strides=pool)(down)
    else:
        x = down
    return (x, down)




"""
def morph_model(input_shape=(None,None, 1)):
    model = Sequential()
    model.add(Erosion2D(1, (10,10),padding="same",input_shape=input_shape))
    model.add(Erosion2D(1, (10,10),padding="same"))
    model.add(Erosion2D(1, (10,10),padding="same"))
    model.add(Dilation2D(1, (10,10),padding="same"))
    model.add(Dilation2D(1, (10,10),padding="same"))
    model.add(Dilation2D(1, (10,10),padding="same"))
    return model
"""

#"""
def morph_model(input_shape=(None,None, 1)):
    model = Sequential()
    model.add(Erosion2D(1, (20,20),padding="same",input_shape=input_shape))
    #model.add(Erosion2D(1, (10,10),padding="same"))
    # model.add(Erosion2D(1, (10,10),padding="same"))
    # model.add(Dilation2D(1, (10,10),padding="same"))
    #model.add(Dilation2D(1, (10,10),padding="same"))
    model.add(Dilation2D(1, (20,20),padding="same"))
    #model.add(Activation("sigmoid"))
    return model
#"""





def morph_model_new(input_shape=(None,None, 1)):
    model = Sequential()
    model.add(Erosion2D(1, (10,10),padding="same",input_shape=input_shape))
    #model.add(Dilation2D(1, (10,10),padding="same",input_shape=input_shape))
    model.add(Dilation2D(1, (10,10),padding="same"))
    #model.add(Erosion2D(1, (10,10),padding="same"))
    model.add(Erosion2D(1, (10,10),padding="same"))
    # model.add(Dilation2D(1, (10,10),padding="same"))
    # model.add(Dilation2D(1, (10,10),padding="same"))
    model.add(Dilation2D(1, (10,10),padding="same"))
    return model






