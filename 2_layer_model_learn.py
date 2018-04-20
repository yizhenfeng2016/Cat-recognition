#coding:utf-8
__author__ = 'Administrator'

import h5py
import numpy as np
import matplotlib.pyplot as plt
from modelmanage import model

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

def predict_y(x,params):
    W1=params["W1"]
    b1=params["b1"]
    W2=params["W2"]
    b2=params["b2"]

    #测试用例样本数
    test_m=x.shape[1]
    prediction_Y=np.zeros((1,test_m))
    #计算
    A1,Z1=model.l_layer_forward_model(x,W1,b1,acitvation="relu") #第1层
    A2,Z2=model.l_layer_forward_model(A1,W2,b2,acitvation="sigmoid") #第2层
    test_Y=A2
    # print(test_Y.shape)
    for i in range(test_Y.shape[1]):
        if test_Y[0][i]>0.5:
            prediction_Y[0][i]=1
        else:
            prediction_Y[0][i]=0

    return prediction_Y

if __name__=="__main__":
    train_x, train_y, test_x, test_y, list_classes=load_dataset()
    #归一化
    train_x=train_x/255
    test_x=test_x/255
    #参数初始化
    params=model.params_init_model(train_x.shape[0],2,[5,1],init_method="He")
    W1=params["W1"]
    b1=params["b1"]
    W2=params["W2"]
    b2=params["b2"]
    learning_rate=0.01
    echo_num=5000
    for echo in range(echo_num):
        #正向传播
        A1,Z1=model.l_layer_forward_model(train_x,W1,b1,acitvation="relu") #第1层
        A2,Z2=model.l_layer_forward_model(A1,W2,b2,acitvation="sigmoid") #第2层

        #计算代价
        J,dJ=model.cost_model(A2,train_y)
        #反向传播
        dA2_prew,dW2,db2=model.l_layer_backward_model(A1,W2,Z2,dJ,acitvation="sigmoid")#第2层
        dA1_prew,dW1,db1=model.l_layer_backward_model(train_x,W1,Z1,dA2_prew,acitvation="relu")#第1层
        #更新参数权值
        # print("w",W.shape)
        # print("dw",dW.shape)
        W1=W1-learning_rate*dW1
        b1=b1-learning_rate*db1

        W2=W2-learning_rate*dW2
        b2=b2-learning_rate*db2

        print("echo {0} cost:{1}".format(echo,J))

    params["W1"]=W1
    params["b1"]=b1
    params["W2"]=W2
    params["b2"]=b2
    train_Y_hat=predict_y(train_x,params)
    print("train_Y_hat:",train_Y_hat.shape)
    print("train_y:",train_y.shape)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(train_Y_hat-train_y)) * 100))

    test_Y_hat=predict_y(test_x,params)
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

    my_image_y=predict_y(my_image_x,params)
    print(my_image_y)
    # plt.plot(cost)
    # plt.xlabel('iter times')
    # plt.ylabel('cost')
    plt.show()

