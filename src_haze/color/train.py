import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras_contrib.losses import DSSIMObjective
import tensorflow as tf
import numpy as np
from morph_layers import *
from generator import *
from  models import *



#model=create_morph_model_type2()
model=total_model()

our_loss=loss_all
num_epochs=1000


datagen=gen_data(batch_size=4)
#datagen=ImageSequence_NYU()




model.compile(loss=[zero_loss], optimizer="RMSprop")
model.fit_generator(datagen, epochs=num_epochs,steps_per_epoch=20)
model.save_weights("./models/model_type2_w.h5")













"""
####   Test  ##################
datagen=gen_data(batch_size=4)
t1,t2=datagen.next()
t2_out=model.predict(t1)
"""
