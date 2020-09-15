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
    model_path1=path1_old()
    model_path2=path2_old()
    model_path12_new=model_new()
    model_cnn=create_CNN_model()

    model_list=[model_path1,model_path2,model_path12_new,model_cnn]
    return model_list




def get_trained_models(path="./models/"):
    model_list=get_model_list()
    model_name=["new_model_path1.h5","_nwe_model_path2.h5","new_model12_new.h5","new_cnn_model.h5"]

    for i in range(len(model_list)):
        print i
        model_list[i].load_weights(path+model_name[i])

    return model_list






def read_resize_image(file_path,size=(416,416,3)):
    Img = misc.imread(file_path)
    Img = resize(Img, size)

    return Img




def read_files(file_in_dir):

    images_in=[]
    images_out=[]

    
    print"getting all the images..."
    file_list=os.listdir(file_in_dir)
    for f1 in file_list:
        img1=read_resize_image(file_in_dir+f1)
        images_in.append(img1) 

    images_in=np.array(images_in,dtype="float32")

    return images_in




def save_images(X,output_dir="./output/",name="p1",):
    for i in range(X.shape[0]):
        misc.imsave(output_dir+str(i)+name+".png",X[i])#,format="rgb")


def save_all_model_images(input_dir="./in",output_dir="./output/"):
    Names={0:"_1_in",1:"_2_p1",2:"_3_p2",3:"_4_p12",4:"_5_cnn",5:"_6_gt"}


    X=read_files(input_dir)
    save_images(X,name=Names[0],output_dir=output_dir)
    model_list=get_trained_models()


    for i in range(len(model_list)):
        model=model_list[i]
        Y_out=model.predict(X,batch_size=32)
        Y_out=np.clip(Y_out,0,1)
        save_images(Y_out,name=Names[i+1],output_dir=output_dir)






###########MAIN TEST CODE###########################################################################
save_all_model_images(input_dir=sys.argv[1],output_dir=sys.argv[2])























