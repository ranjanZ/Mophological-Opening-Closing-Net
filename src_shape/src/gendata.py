import cv2
import numpy as np 
import matplotlib.pyplot as plt




def check_inersection(X,Y):
    t=np.sum(np.bitwise_and(X.astype("bool"),Y.astype("bool")))
    if(t==0):
      return True
    else:
      return False








def gen_image(inp_shape=(512,512)):
      a=inp_shape[0]
      b=inp_shape[1]

      br=15     #radious       
      sr=6     #radi
      B_cir=10  #number of big circle
      S_cir=10  #of small circle

      X=np.zeros((b,a)) 
      count=0
      while(count<B_cir):
          cx=np.random.randint(0+br,a-br)
          cy=np.random.randint(0+br,b-br)
          X1=cv2.circle(np.zeros((b,a)),(cy,cx),br,1,-1)
          if(check_inersection(X1,X)):
              X=X+X1
              count=count+1
  



      count=0
      X_in=X.copy()
      while(count<S_cir):
          cx=np.random.randint(0+sr,a-sr)
          cy=np.random.randint(0+sr,b-sr)
          X1=cv2.circle(np.zeros((b,a)),(cy,cx),sr,1,-1)
          if(check_inersection(X1,X_in)):
              X_in=X_in+X1
              count=count+1
  
      return X_in,X 



def gen_data(num_sample=10):
    X=[]
    Y=[]
    for i in range(num_sample):
        t1,t2=gen_image()
        X.append(t1)
        Y.append(t2)

    X=np.array(X)
    Y=np.array(Y)
     
    X=X[:,:,:,np.newaxis]
    Y=Y[:,:,:,np.newaxis]
    return(X,Y)



X,Y=gen_data()


