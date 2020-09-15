import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np 
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
import matplotlib.pyplot as plt
from skimage.data import data_dir
from skimage.util import img_as_ubyte
from skimage import io
from skimage.color import rgb2gray
from scipy import misc
from skimage.transform import  resize
from keras_contrib.losses import DSSIMObjective
from models  import *
from skimage.morphology import disk
from keras.utils import Sequence



def apply_closing(X):
    selem = disk(5)
    Y = closing(X, selem)
    return Y

def apply_opening(X,size=7):
    selem = disk(size)
    Y = opening(X, selem)
    return Y

def apply_dilation(X,size=7):
    selem = disk(size)
    Y = dilation(X, selem)
    return Y

def apply_erosion(X):
    selem = disk(5)
    Y = dilation(X, selem)
    return Y


def get_data(path="../data/X/",apply_func=apply_closing):

    X=[]
    Y=[]
    files=os.listdir(path)
    for f in files:
        X_file=misc.imread(path+f)
        X_file = img_as_ubyte(X_file)
        #Y_file=apply_closing(X_file)
        Y_file=apply_func(X_file)
        
        X.append(X_file) 
        Y.append(Y_file) 
    X=np.array(X)/255.0
    Y=np.array(Y)/255.0
    X=X[:,:,:,np.newaxis]
    Y=Y[:,:,:,np.newaxis]
    return(X,Y)



class ImageSequence(Sequence):
    def __init__(self,  batch_size=4, input_size=(416, 416),path="/media/newhd/data/flickr30k_images/flickr30k_images/",fold=0):
        self.num_fold=10
        self.image_seq_path = path 
        self.input_shape = input_size
        self.batch_size = batch_size
        self.epoch = 0
        self.SHAPE_Y = self.input_shape[0]
        self.SHAPE_X = self.input_shape[1]
        self.images = sorted([f for f in os.listdir(self.image_seq_path) if f.endswith(".jpg")])[:400]
        self.idx_L=[len(self.images)*i/10 for i in range(self.num_fold)]
        self.step=len(self.images)/10
        self.fold=fold

    def __len__(self):
        return (100)

    def __getitem__(self, idx):
        end_idx = len(self.images)*9/10
        start_idx = 0
        x_batch = []
        y_batch = []
        for s_idx in range(self.batch_size):
            while(True):
              idx=np.random.randint(end_idx)
              if(idx<self.idx_L[self.fold]+self.step and idx>self.idx_L[self.fold]):
                continue
              else:
                break       

            X_file=misc.imread(self.image_seq_path+self.images[idx])
            X_file=rgb2gray(X_file)
            X_file = img_as_ubyte(X_file)
            #Y_file=apply_dilation(X_file)
            Y_file=apply_opening(X_file)
        
            X_file = resize(X_file, (self.SHAPE_Y, self.SHAPE_X, 1))
            Y_file = resize(Y_file, (self.SHAPE_Y, self.SHAPE_X, 1))
            x_batch.append(X_file)
            y_batch.append(Y_file)

        x_batch = np.array(x_batch, np.float32)
        y_batch = np.array(y_batch, np.float32)

        return (x_batch, y_batch)

    def return_test(self):
        f_idx=self.idx_L[self.fold]
        e_idx=f_idx+self.step 
 
        x_batch = []
        y_batch = []
        for idx in range(f_idx,e_idx,1):
            X_file=misc.imread(self.image_seq_path+self.images[idx])
            X_file=rgb2gray(X_file)
            X_file = img_as_ubyte(X_file)
            Y_file=apply_opening(X_file)
        
            X_file = resize(X_file, (self.SHAPE_Y, self.SHAPE_X, 1))
            Y_file = resize(Y_file, (self.SHAPE_Y, self.SHAPE_X, 1))
            x_batch.append(X_file)
            y_batch.append(Y_file)

        x_batch = np.array(x_batch, np.float32)
        y_batch = np.array(y_batch, np.float32)

        return (x_batch, y_batch)

    def on_epoch_end(self):
        self.epoch += 1










"""
A=ImageSequence()
t1,t2=A.__getitem__(3)
t1,t2=A.return_test()
"""


