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

def unet_up_1(filter_count, inputs, down_link, activation='relu', n_layers=3):
    reduced = Conv2D(filter_count, (1, 1), padding='same', activation=activation)(inputs)
    up = UpSampling2D((2, 2))(reduced)
    up = BatchNormalization()(up)
    link = Conv2D(filter_count, (1, 1), padding='same', activation=activation)(down_link)
    link = BatchNormalization()(link)
    up = Add()([up,link])
    for i in range(n_layers):
        up = Conv2D(filter_count, (3, 3), padding='same', activation=activation)(up)
        up = BatchNormalization()(up)
    return up



def create_CNN_model(input_shape=(512,512,3)):

    n_layers_down = [2, 3, 3, 3, 3, 3]
    n_layers_up = [2, 3, 3, 3, 3, 3]
    n_filters_down = [16,32,64, 96, 144, 192]
    n_filters_up = [16,32,64, 96, 144, 192]
    n_filters_center=256
    n_layers_center=4
    print('n_filters_down:%s  n_layers_down:%s'%(str(n_filters_down),str(n_layers_down)))
    print('n_filters_center:%d  n_layers_center:%d'%(n_filters_center, n_layers_center))
    print('n_filters_up:%s  n_layers_up:%s'%(str(n_filters_up),str(n_layers_up)))
    activation='relu'
    inputs = Input(shape=input_shape)
    x = inputs
    x = BatchNormalization()(x)
    xbn = x
    depth = 0
    back_links = []
    for n_filters in n_filters_down:
        n_layers = n_layers_down[depth]
        x, down_link = unet_down_1(n_filters, x, activation=activation, n_layers=n_layers)
        back_links.append(down_link)
        depth += 1

    center, _ = unet_down_1(n_filters_center, x, activation=activation, pool=None, n_layers=n_layers_center)


    # center
    x1 = center
    while depth > 0:
        depth -= 1
        link = back_links.pop()
        n_filters = n_filters_up[depth]
        n_layers = n_layers_up[depth]
        x1 = unet_up_1(n_filters, x1, link, activation=activation, n_layers=n_layers)
        if depth <= 1:
            x1 = Dropout(0.25)(x1)

    x1 = concatenate([x1,xbn])
    x1 = Conv2D(16, (3, 3), padding='same', activation=activation)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(16, (3, 3), padding='same', activation=activation)(x1)
    x1 = BatchNormalization()(x1)

    x1 = Conv2D(3, (1, 1), activation='sigmoid')(x1)
    model = Model(inputs=inputs, outputs=x1)
    return model





#single path of dialtion and erosion lastly linear combination 
def create_morph_model_type2(input_shape=(512, 512, 3)):
    I = Input(shape=input_shape)
    z = I
    for i in range(2):
        z1 = Dilation2D(4, (8, 8), padding="same", strides=(1, 1))(z)
        z2 = Erosion2D(4, (8, 8), padding="same", strides=(1, 1))(z)
        z=Concatenate()([z1,z2])


    #T path
    t=z
    for i in range(2):
        z1 = Dilation2D(4, (8, 8), padding="same", strides=(1, 1))(t)
        z2 = Erosion2D(4, (8, 8), padding="same", strides=(1, 1))(t)
        t=Concatenate()([z1,z2])
    T=Conv2D(1,(1,1),activation="sigmoid")(t)


    #K path
    k=z
    for i in range(2):
        z1 = Dilation2D(4, (8, 8), padding="same", strides=(1, 1))(k)
        z2 = Erosion2D(4, (8, 8), padding="same", strides=(1, 1))(k)
        k=Concatenate()([z1,z2])
    K=Conv2D(3,(1,1),activation="sigmoid")(k)

    model = Model(inputs=[I], outputs=[T,K])
    #model.compile(loss=DSSIMObjective(kernel_size=100), optimizer="RMSprop",metrics=['mse'])
    return model



#separate path last linear combination 
def create_morph_path12_model(input_shape=(512, 512, 3)):
    I=Input(shape=input_shape)
    z1=I
    z2=I
    for i in range(2):
        for j in range(2):
                z1=Dilation2D(4, (8,8),padding="same",strides=(1,1))(z1)
                z2=Erosion2D(4, (8,8),padding="same",strides=(1,1))(z2)
        for j in range(2):
                z1=Erosion2D(4, (8,8),padding="same",strides=(1,1))(z1)
                z2=Dilation2D(4, (8,8),padding="same",strides=(1,1))(z2)

    z1=Erosion2D(4, (8,8),padding="same",strides=(1,1))(z1)
    z2=Dilation2D(4, (8,8),padding="same",strides=(1,1))(z2)
    
    z=Concatenate()([z1,z2])
    z=Conv2D(3,(1,1),activation="sigmoid",padding="same")(z)
    model=Model(inputs=[I],outputs=[z])
    #model.compile(loss=DSSIMObjective(kernel_size=100), optimizer="RMSprop",metrics=['mse'])
    return model



############################################################################


def dehaze_model():
    I = Input(shape = (None, None, 3))


    ## Path 1

    z1=Dilation2D(4, (8,8),padding="same",strides=(1,1))(I)
    z1=Dilation2D(4, (8,8),padding="same",strides=(1,1))(z1)
    z1=Erosion2D(4, (8,8),padding="same",strides=(1,1))(z1)
    z1=Erosion2D(4, (8,8),padding="same",strides=(1,1))(z1)

    for i in range(1):
        z1=Dilation2D(4, (8,8),padding="same",strides=(1,1))(z1)
        z1=Dilation2D(4, (8,8),padding="same",strides=(1,1))(z1)
        z1=Erosion2D(4, (8,8),padding="same",strides=(1,1))(z1)
        z1=Erosion2D(4, (8,8),padding="same",strides=(1,1))(z1)

    z1=Erosion2D(4, (8,8),padding="same",strides=(1,1))(z1)

    # Path 2

    z2=Erosion2D(4, (8,8),padding="same",strides=(1,1))(I)
    z2=Erosion2D(4, (8,8),padding="same",strides=(1,1))(z2)
    z2=Dilation2D(4, (8,8),padding="same",strides=(1,1))(z2)
    z2=Dilation2D(4, (8,8),padding="same",strides=(1,1))(z2)

    for i in range(1):
        z2=Erosion2D(4, (8,8),padding="same",strides=(1,1))(z2)
        z2=Erosion2D(4, (8,8),padding="same",strides=(1,1))(z2)
        z2=Dilation2D(4, (8,8),padding="same",strides=(1,1))(z2)
        z2=Dilation2D(4, (8,8),padding="same",strides=(1,1))(z2)

    z2=Dilation2D(4, (8,8),padding="same",strides=(1,1))(z2)


    z3=Concatenate()([z1, z2])

    z3=Conv2D(8,(1,1),padding='same')(z3)


    # Path for K (m*n*3)

    zk= Dilation2D(4, (8,8),padding="same",strides=(1,1))(z3)
    zk= Erosion2D(4, (8,8),padding="same",strides=(1,1))(zk)
    for i in range(1):
        zk= Dilation2D(4, (8,8),padding="same",strides=(1,1))(zk)
        zk= Erosion2D(4, (8,8),padding="same",strides=(1,1))(zk)


    zk_out = Conv2D(3,(1,1),padding='same',activation='sigmoid')(zk)


    # Path for t (m*n*1)

    zt = Erosion2D(4, (8,8),padding="same",strides=(1,1))(z3)
    zt= Dilation2D(4, (8,8),padding="same",strides=(1,1))(zt)
    for i in range(1):
        zt= Erosion2D(4, (8,8),padding="same",strides=(1,1))(zt)
        zt= Dilation2D(4, (8,8),padding="same",strides=(1,1))(zt)

    zt_out = Conv2D(1,(1,1),padding='same',activation='sigmoid')(zt)



    model=Model(inputs=[I],outputs=[zt_out,zk_out])
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

        """
        loss3 =K.mean(K.abs(J-temp))
        loss2 =K.mean(K.abs(I-J*T-AT))
        loss=loss3+loss2
        """
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
    model=create_morph_model_type2()
    

    J = Input(shape=input_shape)
    I = Input(shape=input_shape)

    T,AT=model(I)
    out=LossLayer_2nd()([I,J,T,AT])
    model_all= Model(inputs=[I,J],outputs=[out])
    return model_all

