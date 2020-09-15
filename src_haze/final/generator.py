# coding=utf-8
import skimage.transform
from keras.utils import Sequence
import numpy as np
import cv2
import os
import glob 
from scipy import misc
from skimage.transform import  resize
#from init import *
import random
from PIL import Image
import h5py
import scipy
import scipy.io as io
from scipy.ndimage.filters import gaussian_filter
from matplotlib import pyplot as plt
#from scipy.imageio import imread
from scipy.misc import imread
from skimage import color as cl
import tensorflow as tf
import sys

BATCH_SIZE = 3
WIDTH = 640
HEIGHT = 480


def read_image(file_path):
    Img=misc.imread(file_path)
    Img=rgb2gray(Img)/255.0
    return Img

def normalize(arr):
    arr=arr.astype('float32')
    if arr.max() > 1.0:
        arr/=255.0
    return arr




def rgb2yuv_batch(rgb_im):
    matrix = np.array([[0.29900, 0.58700, 0.11400], [-0.14713, -0.28886, 0.436] , [0.615,-0.51499, -0.10001]],dtype="float32")
    yuv = np.dot(rgb_im,matrix.T)
    return yuv

def yuv2rgb_batch(yuv_im):
    matrix = np.array([[1., 0., 1.13983], [1., -0.39465, -0.58060] , [1.,2.03211, 0]], dtype = "float32")
    rgb = np.dot(yuv_im,matrix.T)
    return rgb

def read_resize_image(file_path,size=(516,516,3)):
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







###########################################



#DATA_PATH_GT = str('/media/newhd/sancha/haze/synthetic/NYU/nyu_gt/')
#DATA_PATH_HAZE = str( '/media/newhd/sancha/haze/synthetic/NYU/nyu_haze/')




class ImageSequence_NYU(Sequence):
    def __init__(self,  batch_size=BATCH_SIZE, input_size=(WIDTH, HEIGHT)):
        #self.image_seq_path=DATA_PATH
        self.input_shape=input_size
        self.batch_size = batch_size
        self.epoch = 0
        self.SHAPE_Y=self.input_shape[0]
        self.SHAPE_X=self.input_shape[1]
        self.haze_img_path="/media/newhd/sancha/haze/synthetic/NYU/nyu_haze/"
        self.gt_img_path="/media/newhd/sancha/haze/synthetic/NYU/nyu_gt/"

    def __len__(self):
        return (1000)

    def create_files(self):
        pass



    def __getitem__(self,idz):
        haze_path = glob.glob(os.path.join(self.haze_img_path,'*.bmp' ))
        gt_path =  glob.glob(os.path.join(self.gt_img_path,'*_Image_.bmp' ))
        haze_path=sorted(haze_path)
        gt_path=sorted(gt_path)

        gt_rgb = []
        haze_rgb = []
        c=0
        while(c<self.batch_size):
            idx=np.random.randint(len(gt_path))


            img_haze = (imread(haze_path[idx]))/255.0
            img_gt = (imread(gt_path[idx]))/255.0
            img_haze = img_haze[30:-30, 30:-30,:]
            img_gt = img_gt[30:-30, 30:-30,:]

              
            gt_rgb.append(img_gt)
            haze_rgb.append(img_haze)
            #print haze_path[idx],gt_path[idx]
            c=c+1

        gt_rgb=np.array(gt_rgb)
        haze_rgb=np.array(haze_rgb)
        empty_arr  =  np.zeros_like(haze_rgb)

        return [haze_rgb, gt_rgb],empty_arr   


    def on_epoch_end(self):
        self.epoch += 1



class ImageSequence_NTIRE(Sequence):
    def __init__(self,  batch_size=BATCH_SIZE, input_size=(WIDTH, HEIGHT)):
        #self.image_seq_path=DATA_PATH
        self.input_shape=input_size
        self.batch_size = batch_size
        self.epoch = 0
        self.SHAPE_Y=self.input_shape[0]
        self.SHAPE_X=self.input_shape[1]
        self.haze_img_path="/media/newhd/sancha/haze/synthetic/ntire18/outdoor/train_hazy/"
        self.gt_img_path="/media/newhd/sancha/haze/synthetic/ntire18/outdoor/train_gt/"

    def __len__(self):
        return (100)

    def create_files(self):
        pass



    def __getitem__(self,idz):
        haze_path = glob.glob(os.path.join(self.haze_img_path,'*.png' ))
        gt_path =  glob.glob(os.path.join(self.gt_img_path,'*.png' ))
        haze_path=sorted(haze_path)
        gt_path=sorted(gt_path)

        gt_rgb = []
        haze_rgb = []
        c=0
        while(c<self.batch_size):
            idx=np.random.randint(len(gt_path))


            img_haze = (imread(haze_path[idx]))/255.0
            img_gt = (imread(gt_path[idx]))/255.0
            img_haze = img_haze[30:-30, 30:-30,:]
            img_gt = img_gt[30:-30, 30:-30,:]

              
            gt_rgb.append(img_gt)
            haze_rgb.append(img_haze)
            #print haze_path[idx],gt_path[idx]
            c=c+1

        gt_rgb=np.array(gt_rgb)
        haze_rgb=np.array(haze_rgb)
        empty_arr  =  np.zeros_like(haze_rgb)

        return [haze_rgb, gt_rgb],empty_arr   


    def on_epoch_end(self):
        self.epoch += 1




def load_file_list_nyu(haze_img_path="/media/newhd/sancha/haze/synthetic/NYU/nyu_haze/",gt_img_path="/media/newhd/sancha/haze/synthetic/NYU/nyu_gt/"):
    haze_file_list = glob.glob(os.path.join(haze_img_path,'*.bmp' ))
    gt_file_list =  glob.glob(os.path.join(gt_img_path,'*_Image_.bmp' ))
    haze_file_list=sorted(haze_file_list)
    gt_file_list=sorted(gt_file_list)

    return (haze_file_list,gt_file_list)


def load_file_list_fattal(haze_img_path="./data/input/fattal_db/haze/",gt_img_path="./data/input/fattal_db/true/"):
    haze_file_list = glob.glob(os.path.join(haze_img_path,'*.png' ))
    gt_file_list =  glob.glob(os.path.join(gt_img_path,'*.png' ))
    haze_file_list=sorted(haze_file_list)
    gt_file_list=sorted(gt_file_list)

    return (haze_file_list,gt_file_list)


def load_file_list_ntire(haze_img_path="/media/newhd/sancha/haze/synthetic/ntire18/outdoor/train_hazy/",gt_img_path="/media/newhd/sancha/haze/synthetic/ntire18/outdoor/train_gt/"): 
   
    #def load_file_list_ntire(haze_img_path="/media/newhd/sancha/haze/synthetic/ntire19/train_hazy/",gt_img_path="/media/newhd/sancha/haze/synthetic/ntire19/train_gt/"):
    haze_path = glob.glob(os.path.join(haze_img_path,'*.jpg' ))
    gt_path =  glob.glob(os.path.join(gt_img_path,'*.jpg' ))
    #haze_path=os.listdir(haze_img_path)
    #gt_path=os.listdir(haze_img_path)
    haze_file_list=sorted(haze_path)
    gt_file_list=sorted(gt_path)

    return (haze_file_list,gt_file_list)


def parse_function(file_in, file_out,input_shape=(512,512), ext='nyu'):
    image_string_in = tf.read_file(file_in)
    image_string_out = tf.read_file(file_out)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    if(ext=='nyu'):
        image_in = tf.image.decode_bmp(image_string_in, channels=3)
        image_out = tf.image.decode_bmp(image_string_out, channels=3)
    elif(ext=="fattal"):
        image_in = tf.image.decode_png(image_string_in, channels=3)
        image_out = tf.image.decode_png(image_string_out, channels=3)
    else:
        image_in = tf.image.decode_jpeg(image_string_in, channels=3)
        image_out = tf.image.decode_jpeg(image_string_out, channels=3)

    # This will convert to float values in [0, 1]
    image_in = tf.image.convert_image_dtype(image_in, tf.float32)
    image_out = tf.image.convert_image_dtype(image_out, tf.float32)

    image_in = tf.image.resize_images(image_in, input_shape)
    image_out = tf.image.resize_images(image_out, input_shape)

    empty_arr = tf.zeros_like(image_out)
    return image_in, image_out,empty_arr





def  gen_data(batch_size,input_size=(512, 512)):
    #file_in,file_out=load_file_list_ntire()
    file_in,file_out=load_file_list_nyu()
    #file_in,file_out=load_file_list_fattal(haze_img_path="./data/input/fattal_db/haze/",gt_img_path="./data/input/fattal_db/true/")

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
        #yield tf.Session().run(next_batch)
        t1,t2,t3= tf.Session().run(next_batch)
        yield [t1,t2],t3
    #"""


########Test###########
 
#--------testing----------------
"""
datagen=gen_data(batch_size=4)
t1,t2=datagen.next()






from generator import *
dataset=gen_data(batch_size=3)
iter = dataset.make_one_shot_iterator()
el = iter.get_next()

with tf.Session() as sess:
    t1=sess.run(el)


"""




















