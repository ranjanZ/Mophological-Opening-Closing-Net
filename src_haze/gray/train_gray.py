import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Sequential, Model, load_model
from keras_contrib.losses import DSSIMObjective
import tensorflow as tf
import numpy as np
from morph_layers import *
from generator_gray import *
from  models_gray import *
from keras.layers import *



def save_models(model_list,path="./models/"):
    model_name=["1model_path1.h5","1model_path2.h5","1model_path12.h5","1model_cnn.h5","1model12_new.h5","1model_morph_type2.h5"]
    for i in range(len(model_list)):
        model_list[i].save_weights(path+model_name[i])

def load_weights(model_list,path="./models/"):
    model_name=["model_path1.h5","model_path2.h5","model_path12.h5","model_cnn.h5","model12_new.h5","model_morph_type2.h5"]

    for i in range(len(model_list)):
        model_list[i].load_weights(path+model_name[i])
    
    return model_list


def get_model_list(input_shape=(512, 512, 1)):
    model_cnn=create_CNN_model(input_shape=input_shape)
    model_path1=path1_old(input_shape=input_shape)
    model_path2=path2_old(input_shape=input_shape)
    model_path12=path12_old(input_shape=input_shape)
    model_path12_new=create_morph_path12_model(input_shape=input_shape)
    model_morph_type2=create_morph_model_type2(input_shape=input_shape)

    model_list=[model_path1,model_path2,model_path12,model_cnn,model_path12_new,model_morph_type2]
    return model_list



############## Training ##############


"""
model_cnn=create_CNN_model()
model_path1=path1_old()
model_path2=path2_old()
model_path12=path12_old()
model_path12_new=create_morph_path12_model()
model_morph_type2=create_morph_model_type2()

model_list=[model_path1,model_path2,model_path12,model_cnn,model_path12_new,model_morph_type2]

#[model_path1,model_path2,model_path12,model_cnn,model_path12_new,model_morph_type2]=load_weights(model_list)
"""


model_list=get_model_list(input_shape=(480,640,1))
model_list=load_weights(model_list)





#our_loss=DSSIMObjective(kernel_size=23)
our_loss=loss_all
num_epochs=300


#"""
for i in range(len(model_list)):
    datagen=gen_data(batch_size=4)
    model_list[i].compile(loss=our_loss, optimizer="RMSprop",metrics=['mse',DSSIMObjective(kernel_size=23)])
    model_list[i].fit_generator(datagen, epochs=num_epochs,steps_per_epoch=20)

#"""

"""
i=3
datagen=gen_data(batch_size=4)
model_list[i].compile(loss=our_loss, optimizer="RMSprop",metrics=['mse',DSSIMObjective(kernel_size=23)])
model_list[i].fit_generator(datagen, epochs=num_epochs,steps_per_epoch=20)
"""


save_models(model_list)















"""
####   Test  ##################
datagen=gen_data(batch_size=4)
t1,t2=datagen.next()
t2_out=model.predict(t1)
"""
