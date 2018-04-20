#coding:utf-8
__author__ = 'Administrator'

import h5py
import numpy as np
import matplotlib.pyplot as plt
from modelmanage import model
import os

def load_dataset():
    """
    # 加载数据集
    """
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")  # 读取H5文件

    # for t in train_dataset:
    #     print(t)
    # set_x=train_dataset["train_set_y"]
    # print(set_x.shape)
    # list_classes=train_dataset["list_classes"]
    # print(list_classes.shape)
    # for l in list_classes:
    #     print(l)
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    # print(train_set_x_orig.shape)
    # print(train_set_y_orig.shape)
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))  # 对训练集和测试集标签进行reshape
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))


    #矩阵转置
    train_x=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
    test_x=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
    train_y=train_set_y_orig.reshape(train_set_y_orig.shape[1],1).T
    test_y=test_set_y_orig.reshape(test_set_y_orig.shape[1],1).T
    list_classes=classes.reshape(classes.shape[0],1).T
    return train_x, train_y, test_x, test_y, list_classes

def save_params(params):
    cur_filepath=os.path.basename(os.path.realpath(__file__))
    cur_file,ext=os.path.splitext(cur_filepath)
    filename=cur_file+".h5"
    f=h5py.File(filename,'w')
    for k,v in params.items():
        f.create_dataset(k,data=v)

def predict_y_with_load_params(x):
    cur_filepath=os.path.basename(os.path.realpath(__file__))
    cur_file,ext=os.path.splitext(cur_filepath)
    filename=cur_file+".h5"
    f=h5py.File(filename,'r')
    params={}
    for k in f.keys():
        params[k]=np.array(f[k][:])
    W1=params["W1"]
    b1=params["b1"]
    W2=params["W2"]
    b2=params["b2"]
    W3=params["W3"]
    b3=params["b3"]
    W4=params["W4"]
    b4=params["b4"]
    #测试用例样本数
    test_m=x.shape[1]
    prediction_Y=np.zeros((1,test_m))
    #计算
    A1,Z1=model.l_layer_forward_model(x,W1,b1,acitvation="relu") #第1层
    A2,Z2=model.l_layer_forward_model(A1,W2,b2,acitvation="relu") #第2层
    A3,Z3=model.l_layer_forward_model(A2,W3,b3,acitvation="relu") #第3层
    A4,Z4=model.l_layer_forward_model(A3,W4,b4,acitvation="sigmoid") #第4层
    test_Y=A4
    # print(test_Y.shape)
    for i in range(test_Y.shape[1]):
        if test_Y[0][i]>0.5:
            prediction_Y[0][i]=1
        else:
            prediction_Y[0][i]=0

    return prediction_Y

if __name__=="__main__":
    train_X, train_Y, test_x, test_y, list_classes=load_dataset()
    #归一化
    train_X=train_X/255
    # print(train_X.shape)
    test_x=test_x/255
    #参数初始化
    params=model.params_init_model(train_X.shape[0],4,[20,7,5,1],init_method="He",seed=0)
    v,s=model.init_adam(params)
    learning_rate=0.0075
    # lambd=0.7
    echo_num=1000
    grad={}
    for echo in range(echo_num):
        #正向传播
        A1,Z1=model.l_layer_forward_model(train_X,params["W1"],params["b1"],acitvation="relu") #第1层
        A2,Z2=model.l_layer_forward_model(A1,params["W2"],params["b2"],acitvation="relu") #第2层
        A3,Z3=model.l_layer_forward_model(A2,params["W3"],params["b3"],acitvation="relu") #第3层
        A4,Z4=model.l_layer_forward_model(A3,params["W4"],params["b4"],acitvation="sigmoid") #第4层

        #计算代价
        J,dJ=model.cost_model(A4,train_Y)
        # J,dJ=model.cost_with_L2_model(A4,train_Y,params,lambd)
        #反向传播
        dA4_prew,dW4,db4=model.l_layer_backward_model(A3,params["W4"],Z4,dJ,acitvation="sigmoid")#第4层
        dA3_prew,dW3,db3=model.l_layer_backward_model(A2,params["W3"],Z3,dA4_prew,acitvation="relu")#第3层
        dA2_prew,dW2,db2=model.l_layer_backward_model(A1,params["W2"],Z2,dA3_prew,acitvation="relu")#第2层
        dA1_prew,dW1,db1=model.l_layer_backward_model(train_X,params["W1"],Z1,dA2_prew,acitvation="relu")#第1层
        #更新参数权值
        # print("w",W.shape)
        # print("dw",dW.shape)
        grad["dW1"]=dW1
        grad["db1"]=db1
        grad["dW2"]=dW2
        grad["db2"]=db2
        grad["dW3"]=dW3
        grad["db3"]=db3
        grad["dW4"]=dW4
        grad["db4"]=db4
        params,v,s=model.update_params_with_adam(params,grad,learning_rate,v,s,t=2)
        print("echo {0} cost:{1}".format(echo,J))

    save_params(params)
    # train_Y_hat=predict_y(train_x,params)
    train_Y_hat=predict_y_with_load_params(train_X)
    print("train_Y_hat:",train_Y_hat.shape)
    print("train_y:",train_Y.shape)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(train_Y_hat-train_Y)) * 100))

    # test_Y_hat=predict_y(test_x,params)
    test_Y_hat=predict_y_with_load_params(test_x)
    print("prediction_Y:",test_Y_hat.shape)
    print("test_y:",test_y.shape)
    print("test accuracy: {} %".format(100 - np.mean(np.abs(test_Y_hat-test_y)) * 100))
    #
    from scipy import ndimage,misc
    from function import imagefunc
    image_path="/my_image.jpg"
    frame="images"+image_path
    my_image_x=np.array(ndimage.imread(frame,flatten=False))
    print(my_image_x.shape)
    my_image_x=misc.imresize(my_image_x,size=(64,64))
    plt.imshow(my_image_x)
    print(my_image_x.shape)
    my_image_x=my_image_x/255
    my_image_x=imagefunc.image2vector(my_image_x)
    print(my_image_x.shape)

    # my_image_y=predict_y(my_image_x,params)
    my_image_y=predict_y_with_load_params(my_image_x)
    print(my_image_y)
    # plt.plot(cost)
    # plt.xlabel('iter times')
    # plt.ylabel('cost')
    plt.show()



