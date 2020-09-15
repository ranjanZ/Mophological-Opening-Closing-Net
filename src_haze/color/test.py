import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



import tensorflow as tf
import numpy as np
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from morph_layers import *
from generator import *
from  models import *




def get_model_list():
    model_morph_type2=create_morph_model_type2()

    model_list=[model_morph_type2]
    return model_list




def get_trained_models(path="./models/"):
    model_list=get_model_list()
    model_name=["model_type2_w.h5"]

    for i in range(len(model_list)):
        print i
        model_list[i].load_weights(path+model_name[i])

    return model_list






def read_resize_image(file_path):
    Img = misc.imread(file_path)
    print Img.shape
    #Img = rgb2gray(Img)/255.0
    Img = resize(Img, (512,512,3))

    return Img




def read_files(file_in,file_out):

    images_in=[]
    images_out=[]
    for f1,f2 in zip(file_in,file_out)[:5]:
        img1=read_resize_image(f1)
        img2=read_resize_image(f2)
        images_in.append(img1) 
        images_out.append(img2) 

    images_in=np.array(images_in,dtype="float32")
    images_out=np.array(images_out,dtype="float32")

    return images_in,images_out



def calculate_score(Y_out,Y_gt):

    Score=[]
    print("computing  Score")
    for i in range(Y_out.shape[0]):
        t1=psnr(Y_out[i],Y_gt[i])
        t2=ssim(Y_out[i],Y_gt[i],multichannel=True)
        Score.append([t1,t2])

    Score=np.array(Score)
    Score=np.mean(Score,axis=0)
    return  Score   




###########################################################################


model_list=get_trained_models()


file_in,file_out=get_in_out_file_test()
X,Y_gt=read_files(file_in,file_out)

S=[]
for model in model_list:
    Y_out=model.predict(X)
    Y_out=np.clip(Y_out,0,1)
    score=calculate_score(Y_out,Y_gt)
    S.append(score)




