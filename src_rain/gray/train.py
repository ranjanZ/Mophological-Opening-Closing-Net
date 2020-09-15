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


"""
def save_models(model_list,path="./models/"):
    model_name=["model_path1.h5","model_path2.h5","model_path12.h5","model_cnn.h5","model12_new.h5","model_morph_type2.h5"]
    for i in range(len(model_list)):
        model_list[i].save_weights(path+model_name[i])

"""
def load_weights(model_list,path="./models/"):
    #model_name=["model_path1.h5","model_path2.h5","model_path12.h5","model_cnn.h5","model12_new.h5","model_morph_type2.h5"]

    model_name=["new_model_path1.h5","_nwe_model_path2.h5","new_model12_new.h5","new_cnn_model.h5"]
    for i in range(len(model_list)):
        model_list[i].load_weights(path+model_name[i])
    
    return model_list

def get_model_list():
    model_cnn=create_CNN_model()
    model_path1=path1_old()
    model_path2=path2_old()
    #model_path12=path12_old()
    #model_path12_new=create_morph_path12_model()
    #model_morph_type2=create_morph_model_type2()

    model_list=[model_path1,model_path2,model_path12,model_cnn,model_path12_new,model_morph_type2]
    return model_list


def save_models(model_list,path="./models/"):
    model_name=["new_model_path1.h5","_nwe_model_path2.h5","new_model12_new.h5","new_cnn_model.h5"]
    for i in range(len(model_list)):
        model_list[i].save_weights(path+model_name[i])




file_in,file_out=get_in_out_file()
X,Y_gt=read_files(file_in,file_out)
model_path1=path1_old()
model_path2=path2_old()
model_path12_new=model_new()
model_cnn=create_CNN_model()


model_list=[model_path1,model_path2,model_path12_new,model_cnn]
model_list=load_weights(model_list)



#our_loss=DSSIMObjective(kernel_size=23)
#our_loss=loss_all
num_epochs=1

for i in range(len(model_list)):
    model_list[i].compile(loss=DSSIMObjective(kernel_size=23), optimizer="RMSprop",metrics=['mse',DSSIMObjective(kernel_size=23)])
    model_list[i].fit(X,Y_gt, epochs=num_epochs,batch_size=4)


#save_models(model_list)















"""
####   Test  ##################
datagen=gen_data(batch_size=4)
t1,t2=datagen.next()
t2_out=model.predict(t1)
"""
