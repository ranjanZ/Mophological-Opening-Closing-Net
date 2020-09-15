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
model,dh_model=total_model(input_shape=(None,None,3))
model.load_weights("./models/total_model.h5")
dh_model.load_weights("./models/dh_model.h5")



num_epochs=10000


datagen=gen_data(batch_size=4)
model.compile(loss=[zero_loss], optimizer="RMSprop")
model.fit_generator(datagen, epochs=num_epochs,steps_per_epoch=20)



model.save_weights("./models/total_model.h5")
dh_model.save_weights("./models/dh_model.h5")













"""
####   Test  ##################
datagen=gen_data(batch_size=4)
t1,t2=datagen.next()
t2_out=model.predict(t1)

T,AT=dh_model.predict(t1)


"""








