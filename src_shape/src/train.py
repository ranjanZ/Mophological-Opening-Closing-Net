import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from models import *
from gendata import *
from scipy import misc
from matplotlib.ticker import FormatStrFormatter
from keras_contrib.losses import DSSIMObjective
from generator import *


def dis_st(model,num_s):

    for i in range(len(model.layers[:2])):
        l=model.layers[i]
        w=l.get_weights()[0]
        s="../images/"+str(num_s)+"_"+str(i)+".jpg"
        misc.imsave(s,w[:,:,0,0])
        #plt.figure(i)
        #plt.imshow(w[:,:,0,0],cmap="gray")



def loss_f(y_true,y_gt):
    l=K.abs(y_true-y_gt)+K.square(y_true-y_gt)
    loss=K.mean(l)
    return loss


def train_model(model,num_epochs=100,data=(X,Y)):
    model.compile(loss="mse", optimizer="RMSprop")
    #model.compile(loss="sparse_categorical_crossentropy", optimizer="RMSprop")
    #model.compile(loss=DSSIMObjective(kernel_size=200), optimizer="RMSprop")
    model.fit(X,Y,epochs=num_epochs,batch_size=20)
    return model




def test_properties(model,data):

    X,Y=data
    #idempotent error
    Y1=model.predict(X)
    Y2=model.predict(Y1)

    ID=np.mean(np.abs(Y2-Y1),axis=(0,1,2,3))




    #anti extensive  error
    M=(Y1>X)
    E=np.mean(M*(Y1-X),axis=(0,1,2,3))

    
    #increasing
    X1=X
    X2=np.clip(X1+np.random.random(X.shape)*0.5,0,1)
    X1_out=model.predict(X1)
    X2_out=model.predict(X2)

    M=(X1<=X2)
    In=np.mean(np.abs(M*(X1_out-X2_out)),axis=(0,1,2,3))
    
    return (ID,E,In)
    

def test_properties_each_point(model,data):

    X,Y=data
    #idempotent error
    Y1=model.predict(X)
    Y2=model.predict(Y1)
    Y1=np.clip(Y1,0,1)
    Y2=np.clip(Y2,0,1)

    ID=np.mean(np.abs(Y2-Y1),axis=(1,2,3))




    #anti extensive  error
    M=(Y1>X)
    E=np.mean(M*(Y1-X),axis=(1,2,3))

    
    #increasing
    X1=X
    X2=np.clip(X1+np.random.random(X.shape)*0.5,0,1)
    X1_out=model.predict(X1)
    X2_out=model.predict(X2)

    X1_out=np.clip(X1_out,0,1)
    X2_out=np.clip(X2_out,0,1)

    M1=(X2>=X1)
    In_t=(M1*(X2_out-X1_out))
    M2=(In_t<=0)
    In=np.mean(np.abs(M2*In_t),axis=(1,2,3))
    return (ID,E,In)
    

def plot(L,num_sample_L):
    name="../images/plot_prop_error.pdf"
    lgd=["Idempotent Error","Anti Extensive  Error"]
    fig=plt.figure("ERROR Plot")
    ax=plt.gca()
   
    plt.plot(num_sample_L,L[:,0],linewidth=2)
    plt.plot(num_sample_L,L[:,1],linewidth=2)
    plt.rc('text', usetex=True)
    plt.legend(lgd, loc='upper right',fontsize=20)
    plt.xlabel("Number of Sample",fontsize=25,weight=None)
    plt.ylabel("Error",fontsize=29,weight=None)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=20)
    plt.grid(True,color="k",alpha=0.2)
    fig.subplots_adjust(left=0.147, bottom=0.130, right=0.994, top=0.990)
    fig.set_size_inches(7.7, 6.2)
    fig.savefig(name)
    fig.savefig(name.replace(".pdf",".svg"))

    #fig.clear()









def run_mutiple():

    L=[] 
    #for num_s in [50,100,200,400,800]:
    num_sample_L=[40,80,100,200,300,400]
    for num_s in num_sample_L:
        X,Y=gen_data(num_sample=num_s)
        idx=X.shape[0]*9/10
        X_train=X[:idx]
        Y_train=Y[:idx]
        X_test=X[idx:]
        Y_test=Y[idx:]

        model=morph_model()
        model=train_model(model,num_epochs=400,data=(X_train,Y_train))
        model.save_weights("../data/models_"+str(num_s)+".h5")
        M=test_properties(model,data=(X_test,Y_test))
        L.append(M)
        #dis_st(model,num_s)


def run_single():

    num_s=400
    X,Y=gen_data(num_sample=num_s)
    step=X.shape[0]/10
    idx_L=[X.shape[0]*i/10 for i in range(10)]
    L=[]
    for idx in idx_L:
      #print idx,idx+step

      X_train=np.concatenate((X[:idx],X[idx+step:]),axis=0)
      Y_train=np.concatenate((Y[:idx],Y[idx+step:]),axis=0)
      X_test=X[idx:idx+step]
      Y_test=Y[idx:idx+step]
      print X_train.shape,X_test.shape
      model=morph_model(input_shape=(512,512,1))
      #model.load_weights("../data/models_"+str(idx)+".h5")
      model=train_model(model,num_epochs=800,data=(X_train,Y_train))
      #model.save_weights("../data/models_"+str(idx)+".h5")
      M=test_properties_each_point(model,data=(X_test,Y_test))
      L.append(M)
      #dis_st(model,idx)

    L=np.array(L)
    L=np.swapaxes(L,-2,-1)
    S=L.reshape((-1,3))
    plt.figure()
    #plt.hist(S[:,0],100,range=[0,0.3])
    plt.hist(S[:,0],100)
    plt.savefig("../images/Id_error.pdf")
    plt.figure()
    plt.hist(S[:,1],100,range=[0,1])
    #plt.hist(S[:,1],100)
    #plt.savefig("../images/Ext_error.pdf")
    plt.figure()
    plt.hist(S[:,2],100,range=[0,1])
    #plt.hist(S[:,2],100)
    plt.savefig("../images/In_error.pdf")


#f1=ssim
#f2=mse
def run_new():
    L1=[]
  
    for f in range(10):
      imgen=ImageSequence(fold=f)
      model=morph_model(input_shape=(416,416,1))
      model.compile(loss="mse", optimizer="RMSprop")
      #model.compile(loss=DSSIMObjective(kernel_size=23), optimizer="RMSprop")
      model.load_weights("../data/f2models_"+str(f)+".h5")
      #model.fit_generator(imgen,epochs=10)
      #model.save_weights("../data/f2models_"+str(f)+".h5")
      dis_st(model,f)

      X_test,Y_test=imgen.return_test()
      M=test_properties_each_point(model,data=(X_test,Y_test))
      L1.append(M)

    L=np.array(L1)
    L=np.swapaxes(L,-2,-1)
    S=L.reshape((-1,3))
    S_mean=np.mean(S,axis=0)
    S_std=np.std(S,axis=0)
    plt.figure()
    plt.hist(S[:,0],100,range=[0,1])
    #plt.hist(S[:,0],100)
    plt.savefig("../images/Id_error.pdf")
    plt.figure()
    plt.hist(S[:,1],100,range=[0,1])
    #plt.hist(S[:,1],100)
    plt.savefig("../images/Ext_error.pdf")
    plt.figure()
    plt.hist(S[:,2],100,range=[0,1])
    #plt.hist(S[:,2],100)
    plt.savefig("../images/In_error.pdf")





