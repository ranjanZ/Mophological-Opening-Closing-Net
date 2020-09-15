#####################################################
import numpy as np
#from  simple_NN import *
#from tensorflow.examples.tutorials.mnist import input_data
import os
import re
import pandas
import matplotlib.patches as mpatches
from keras_contrib.losses import DSSIMObjective
import keras.backend as K
import tensorflow as tf
#from keras.objectives import *
import keras_contrib.backend as KC

# coding=utf-8
from keras import Input, Model
from keras.layers import *
from morph_layers import *



############################################################################


def dehaze_model():
    I = Input(shape = (None, None, 3))

    #I=Input(shape=input_shape)
    z1=I
    z2=I
    for i in range(2):
        for j in range(1):
                z1=Dilation2D(4, (4,4),padding="same",strides=(1,1))(z1)
                z2=Erosion2D(4, (4,4),padding="same",strides=(1,1))(z2)
        for j in range(1):
                z1=Erosion2D(4, (4,4),padding="same",strides=(1,1))(z1)
                z2=Dilation2D(4, (4,4),padding="same",strides=(1,1))(z2)
    z=Concatenate()([z1,z2])
    z=Conv2D(3,(1,1),activation="linear",padding="same")(z)


    z1=z
    z2=z
    for i in range(4):
        for j in range(1):
                z1=Dilation2D(4, (4,4),padding="same",strides=(1,1))(z1)
                z2=Erosion2D(4, (4,4),padding="same",strides=(1,1))(z2)
        for j in range(1):
                z1=Erosion2D(4, (4,4),padding="same",strides=(1,1))(z1)
                z2=Dilation2D(4, (4,4),padding="same",strides=(1,1))(z2)
    T=Concatenate()([z1,z2])
    T=Conv2D(1,(1,1),activation="sigmoid",padding="same")(T)




    z1=z
    z2=z
    for i in range(4):
        for j in range(1):
                z1=Dilation2D(4, (4,4),padding="same",strides=(1,1))(z1)
                z2=Erosion2D(4, (4,4),padding="same",strides=(1,1))(z2)
        for j in range(1):
                z1=Erosion2D(4, (4,4),padding="same",strides=(1,1))(z1)
                z2=Dilation2D(4, (4,4),padding="same",strides=(1,1))(z2)
    A=Concatenate()([z1,z2])
    A=Conv2D(3,(1,1),activation="sigmoid",padding="same")(A)



    model=Model(inputs=[I],outputs=[T,A])
    return model































##############################LOSS########################################
def PSNRLoss(y_true, y_pred):
    #return 10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)
    loss=-tf.image.psnr(y_true, y_pred, max_val=1)
    return loss


def total_loss(y_true,y_pred):
    s1 = tf.get_variable("sig1", shape=(1,), trainable=True)
    s2 = tf.get_variable("sig2", shape=(1,), trainable=True)
    loss=(1/(s1*s1))*DSSIMObjective(kernel_size=100)(y_true,y_pred)+ (1/(s2*s2))*(PSNRLoss(y_true,y_pred))+ K.log(s1*s1)+K.log(s2*s2)
    return loss



def total_loss1(y_true,y_pred):
    s1 = tf.get_variable("sig1", shape=(1,), trainable=True)
    s2 = tf.get_variable("sig2", shape=(1,), trainable=True)
    #loss=(1/(s1*s1+K.epsilon()))*DSSIMObjective(kernel_size=100)(y_true,y_pred)+ (1/(s2*s2+K.epsilon()))*K.mean(K.abs(y_true-y_pred))+2*K.log(s1)+2*K.log(s2)
    loss=(1/(s1*s1))*DSSIMObjective(kernel_size=100)(y_true,y_pred)+ (1/(s2*s2))*K.mean(K.abs(y_true-y_pred))+K.log(s1*s1)+K.log(s2*s2)

    return(loss)

def loss_new(y_true,y_pred):
    #loss=DSSIMObjective(kernel_size=20)(y_true,y_pred)+ K.mean(K.abs(y_true-y_pred))
    loss_ssim=DSSIMObjective(kernel_size=100)(y_true,y_pred)   #+ K.mean(K.abs(y_true-y_pred))
    loss_psnr=PSNRLoss(y_true,y_pred)
    loss=loss_ssim+0.006*loss_psnr
    return loss


def loss_all(y_pred,y_true):
    loss_psnr=-tf.image.psnr(y_true, y_pred, max_val=1)
    loss_ssim=(1-tf.image.ssim_multiscale(y_true, y_pred,max_val=1))/2.0
    loss_mse=K.mean(K.square(y_true-y_pred))

    loss=loss_mse+loss_ssim
    return loss

def zero_loss(y_true, y_pred):
    return tf.zeros_like(y_true)




class LossLayer_2nd(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(LossLayer_2nd, self).__init__(**kwargs)


    def concat(self,a1,a2,a3):
        a=K.concatenate([a1,a2,a3],axis=-1)
        J=tf.image.yuv_to_rgb(a)
        return J

    #I and J are in RGB and T and K=A(1-T) 
    def lossfun(self, I,J,T,AT):

        T=K.tile(T,[1,1,1,3])
        J_out=(I-AT)/(T+K.epsilon())
        loss1=(1-tf.image.ssim_multiscale(J,J_out,max_val=1.0))/2.0
        loss2=(1-tf.image.ssim_multiscale(I,J*T+AT,max_val=1.0))/2.0

        #loss3=(1-tf.image.ssim_multiscale(J,J_out,max_val=1.0,filter_size=30))/2.0
        #loss4=(1-tf.image.ssim_multiscale(I,J*T+AT,max_val=1.0,filter_size=30))/2.0



        #loss3 =K.mean(K.abs(J-J_out))
        #loss4 =K.mean(K.square(J-J_out))


        """
        loss3 =K.mean(K.abs(J-temp))
        loss2 =K.mean(K.abs(I-J*T-AT))
        loss=loss3+loss2
        """
        #loss_ext =K.mean(K.abs(I-J*T-AT))
        loss=loss1+loss2

        return loss,J_out

    def call(self, inputs):
        I = inputs[0]
        J = inputs[1]
        T = inputs[2]
        AT = inputs[3]


        loss,J_rgb= self.lossfun(I,J,T,AT)
        self.add_loss(loss, inputs=inputs)

        return J_rgb







def total_model(input_shape=(512,512,3)):
    model=dehaze_model()
    

    J = Input(shape=input_shape)
    I = Input(shape=input_shape)

    T,AT=model(I)
    out=LossLayer_2nd()([I,J,T,AT])
    model_all= Model(inputs=[I,J],outputs=[out])
    return model_all,model

