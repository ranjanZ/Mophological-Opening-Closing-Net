import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



import tensorflow as tf
import numpy as np
import skimage
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from morph_layers import *
from generator import *
from  models import *
from skimage.color import rgb2lab



def get_model():
    model,dh_model=total_model(input_shape=(None,None,3))
    dh_model.load_weights("./models/dh_model.h5")
    return dh_model




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



def dehaze_img(model,img):
    img=img[np.newaxis,:,:,:]

    t,at=model.predict(img)
    J=(img-at)/(t+0.00001)
    J=np.clip(J,0,1)
    return J[0],t[0][:,:,0],at[0]


def calculate_score(output_dir="./data/output/",gt_dir="./data/input/fattal_db/true/",key1="im0",key2="im0"):
    img_files=os.listdir(gt_dir)

    Score=[]
    print("computing  Score")
    s=[]
    for i in range(len(img_files)):
        gt_img = misc.imread(gt_dir+img_files[i])/255.0

        out_f=img_files[i]
        out_f=out_f.replace(key1,key2) 

        out_img = misc.imread(output_dir+out_f)/255.0
        #gt_img = read_resize_image(gt_dir+img_files[i])

        t1=psnr(out_img,gt_img)
        t2=ssim(out_img,gt_img,multichannel=True)
        t3=skimage.color.deltaE_ciede2000(rgb2lab(out_img),rgb2lab(gt_img))
        t3=np.mean(t3,axis=(0,1))
        print t3.shape 
  
        Score.append([t1,t2,t3])
        s.append([out_f,round(t1,2),round(t2,2)])
    Score=np.array(Score)
    mean=np.mean(Score,axis=0)

    return  s,Score,mean


def run_model(input_dir="./",output_dir="./data/output/"):
    model=get_model()
    img_files=os.listdir(input_dir)

    for f1 in img_files:
        print f1
        #img=read_resize_image(input_dir+f1)
        img=misc.imread(input_dir+f1,mode="RGB")/255.0
        img_out,T,AT=dehaze_img(model,img)    
        misc.imsave(output_dir+f1,img_out)
        misc.imsave(output_dir+"T_"+f1,T)
        misc.imsave(output_dir+"K_"+f1,AT)





###########################################################################
D={"fattal":("/media/newhd/sancha/haze/synthetic/fattal_db/haze/","/media/newhd/sancha/haze/synthetic/fattal_db/true/","./data/output/fattal/"),
  "middlebury":("/media/newhd/sancha/haze/synthetic/Middlebury/hazy/","/media/newhd/sancha/haze/synthetic/Middlebury/gt/","./data/output/middlebury/"),
  "ntire18":("/media/newhd/sancha/haze/synthetic/ntire18/outdoor/val_hazy/","/media/newhd/sancha/haze/synthetic/ntire18/outdoor/val_gt/","./data/output/ntire18/"),
  "nyu":("/media/newhd/sancha/haze/synthetic/NYU/nyu_haze/","/media/newhd/sancha/haze/synthetic/NYU/nyu_gt/","./data/output/nyu/")
}
  



#run_model(input_dir=D['ntire18'][0],output_dir=D['ntire18'][2]) 
#run_model(input_dir=D['middlebury'][0],output_dir=D['middlebury'][2]) 
#run_model(input_dir=D['fattal'][0],output_dir=D['fattal'][2]) 



#s,S,mean=calculate_score(output_dir=D['ntire18'][2],gt_dir=D['ntire18'][1])
#s,S,mean=calculate_score(output_dir=D['middlebury'][2],gt_dir=D['middlebury'][1],key1="im0.png",key2="Hazy.bmp")
#s,S,mean=calculate_score(output_dir=D['fattal'][2],gt_dir=D['fattal'][1],key1="true",key2="input")
#print mean



run_model(input_dir=sys.argv[1]) 




