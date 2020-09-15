# coding=utf-8
import os
import sys
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import skimage.transform
import tensorflow as tf
from keras.utils import Sequence
import numpy as np
import cv2
import glob
import pandas as pd
import os
from scipy import misc
from skimage.transform import resize
import keras.backend as K 

DATA_PATH = "/media/newhd/data/RAIN/rainy-image-dataset/"


def rgbf2bgr(rgbf):
    t = rgbf*255.0
    t = np.clip(t, 0., 255.0)
    bgr = t.astype(np.uint8)[..., ::-1]
    return bgr


def rgbf2rgb(rgbf):
    t = rgbf*255.0
    t = np.clip(t, 0., 255.0)
    rgb = t.astype(np.uint8)
    return rgb


def rgb2gray(rgb):

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def read_resize_image(file_path,size=(416,416,3)):
    Img = misc.imread(file_path)
    #print Img.shape
    #Img = rgb2gray(Img)/255.0
    Img = resize(Img, size)

    return Img




def read_files(file_in,file_out):

    images_in=[]
    images_out=[]
    print"getting all the images..."
    i=0
    for f1,f2 in zip(file_in,file_out):
        img1=read_resize_image(f1)
        img2=read_resize_image(f2)
        images_in.append(img1)
        images_out.append(img2)
        i=i+1
        sys.stdout.write("\r %d \n"%(i,))
    images_in=np.array(images_in,dtype="float32")
    images_out=np.array(images_out,dtype="float32")

    return images_in,images_out





class ImageSequence(Sequence):
    def __init__(self,  batch_size=4, input_size=(512, 512)):
        self.image_seq_path = DATA_PATH
        self.input_shape = input_size
        self.batch_size = batch_size
        self.epoch = 0
        self.SHAPE_Y = self.input_shape[0]
        self.SHAPE_X = self.input_shape[1]
        self.frames = sorted(os.listdir(self.image_seq_path+"ground_truth/"))

    def __len__(self):
        return (100)

    def read_image(self, file_path):
        Img = misc.imread(file_path)
        print Img.shape
        #Img = rgb2gray(Img)/255.0
        return Img

    def __getitem__(self, idx):
        x_batch = []
        c = 0
        num_frames = len(self.frames)*9/10
        end_idx = len(self.frames)
        start_idx = num_frames
        while(c < self.batch_size):

            # s_idx =np.random.randint(0,num_frames)  #sence IDX
            s_idx = np.random.randint(start_idx, end_idx)

            I2_file = self.image_seq_path+"ground_truth/"+self.frames[s_idx]
            t1 = np.random.randint(1, 15)

            I1_file = self.image_seq_path+"rainy_image/" + \
                self.frames[s_idx][:-4]+"_"+str(t1)+self.frames[s_idx][-4:]
            I1 = self.read_image(I1_file)
            I2 = self.read_image(I2_file)
            # I1=resize(I1,(self.SHAPE_Y,self.SHAPE_X,3))
            # I2=resize(I2,(self.SHAPE_Y,self.SHAPE_X,3))
            I1 = resize(I1, (self.SHAPE_Y, self.SHAPE_X, 3))
            I2 = resize(I2, (self.SHAPE_Y, self.SHAPE_X, 3))
            # I1=I1[:,:,np.newaxis]
            # I2=I2[:,:,np.newaxis]
            x_batch.append([I1, I2])
            c = c+1

        x_batch = np.array(x_batch, np.float32)
        #x_batch=np.zeros((4,2,240, 360, 1))
        # return ([x_batch[:,0,:,:,:],x_batch[:,1,:,:,:]],y_batch)
        # return (x_batch[:,0,:,:,:],x_batch[:,0,:,:,:]-x_batch[:,1,:,:,:])
        return (x_batch[:, 0, :, :, :], x_batch[:, 1, :, :, :])
        # return x_batch

    def on_epoch_end(self):
        self.epoch += 1

    def get_test_data(self):
        end_idx = len(self.frames)
        start_idx = end_idx*9/10
        x_batch = []
        # for s_idx in range(start_idx,start_idx+5,1):
        for s_idx in range(start_idx, end_idx, 1):
            I2_file = self.image_seq_path+"ground_truth/"+self.frames[s_idx]
            for j in range(1, 15, 1):
                I1_file = self.image_seq_path+"rainy_image/" + \
                    self.frames[s_idx][:-4]+"_"+str(j)+self.frames[s_idx][-4:]
                I1 = self.read_image(I1_file)
                I2 = self.read_image(I2_file)
                I1 = resize(I1, (self.SHAPE_Y, self.SHAPE_X, 1))
                I2 = resize(I2, (self.SHAPE_Y, self.SHAPE_X, 1))
                x_batch.append([I1, I2])
        x_batch = np.array(x_batch, np.float32)
        return (x_batch[:, 0, :, :, :], x_batch[:, 1, :, :, :])


class TestImageSequence(Sequence):
    def __init__(self,  batch_size=4, input_size=(512, 512)):
        self.image_seq_path = DATA_PATH
        self.input_shape = input_size
        self.batch_size = batch_size
        self.epoch = 0
        self.SHAPE_Y = self.input_shape[0]
        self.SHAPE_X = self.input_shape[1]
        self.frames = sorted(os.listdir(self.image_seq_path+"ground_truth/"))

    def __len__(self):
        return (100)

    def __getitem__(self, idx):
        end_idx = len(self.frames)
        start_idx = end_idx*9/10
        x_batch = []
        # for s_idx in range(start_idx,start_idx+1,1):
        for s_idx in range(start_idx, end_idx, 1):
            I2_file = self.image_seq_path+"ground_truth/"+self.frames[s_idx]
            for j in range(1, 15, 1):
                I1_file = self.image_seq_path+"rainy_image/" + \
                    self.frames[s_idx][:-4]+"_"+str(j)+self.frames[s_idx][-4:]
                I1 = read_image(I1_file)
                I2 = read_image(I2_file)
                I1 = resize(I1, (self.SHAPE_Y, self.SHAPE_X, 1))
                I2 = resize(I2, (self.SHAPE_Y, self.SHAPE_X, 1))
                x_batch.append([I1, I2])
        x_batch = np.array(x_batch, np.float32)
        return (x_batch[:, 0, :, :, :], x_batch[:, 1, :, :, :])

    def on_epoch_end(self):
        self.epoch += 1


############Tensorflow data generator###################333

def get_in_out_file(image_seq_path="/media/newhd/data/RAIN/rainy-image-dataset/"):

    output_dir=image_seq_path+"ground_truth/"
    frames=os.listdir(output_dir)
    start_idx = 0
    end_idx= len(frames)*9/10
    x_batch = []

    file_in_L=[]
    file_out_L=[]
    for s_idx in range(start_idx, end_idx, 1):
        out_file = image_seq_path+"ground_truth/"+frames[s_idx]
        for j in range(1, 15, 1):
            in_file = image_seq_path+"rainy_image/" + \
                frames[s_idx][:-4]+"_"+str(j)+frames[s_idx][-4:]
            file_in_L.append(in_file)
            file_out_L.append(out_file)

 
    return file_in_L,file_out_L


def get_in_out_file_test(image_seq_path="/media/newhd/data/RAIN/rainy-image-dataset/"):

    output_dir=image_seq_path+"ground_truth/"
    frames=os.listdir(output_dir)
    end_idx = len(frames)
    start_idx = end_idx*9/10
    x_batch = []

    file_in_L=[]
    file_out_L=[]
    for s_idx in range(start_idx, end_idx, 1):
        out_file = image_seq_path+"ground_truth/"+frames[s_idx]
        for j in range(1, 15, 1):
            in_file = image_seq_path+"rainy_image/" + \
                frames[s_idx][:-4]+"_"+str(j)+frames[s_idx][-4:]
            file_in_L.append(in_file)
            file_out_L.append(out_file)

    return file_in_L,file_out_L




def parse_function(file_in, file_out,input_shape=(512,512)):
    image_string_in = tf.read_file(file_in)
    image_string_out = tf.read_file(file_out)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image_in = tf.image.decode_jpeg(image_string_in, channels=3)
    image_out = tf.image.decode_jpeg(image_string_out, channels=3)

    # This will convert to float values in [0, 1]
    image_in = tf.image.convert_image_dtype(image_in, tf.float32)
    image_out = tf.image.convert_image_dtype(image_out, tf.float32)

    image_in = tf.image.resize_images(image_in, input_shape)
    image_out = tf.image.resize_images(image_out, input_shape)
    return image_in, image_out





def  gen_data(batch_size,input_size=(512, 512)):
    file_in,file_out=get_in_out_file()
    dataset = tf.data.Dataset.from_tensor_slices((file_in, file_out))
    dataset = dataset.shuffle(len(file_in))
    dataset = dataset.map(parse_function, num_parallel_calls=8)
    #dataset = dataset.map(train_preprocess, num_parallel_calls=8)
    dataset = dataset.batch(batch_size)

    #dataset = dataset.repeat()
    dataset = dataset.prefetch(2)
    #return dataset


    #"""
    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()
    while True:
        #yield K.get_session().run(next_batch)
        yield tf.Session().run(next_batch)
    #"""








#--------testing----------------
"""

from generator import *
dataset=gen_data(batch_size=3)
iter = dataset.make_one_shot_iterator()
el = iter.get_next()

with tf.Session() as sess:
    t1=sess.run(el)



"""












"""
from generator import *
A=ImageSequence()
t1,t2=A.__getitem__(3)
t1,t2=A.get_test_data()

"""


"""
from generator import *
A=ImageSequence()
t1,t2=A.__getitem__(3)
t1,t2=A.get_test_data()

"""
