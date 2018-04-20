#coding:utf-8
__author__ = 'Administrator'

#coding:utf-8
__author__ = 'Administrator'

import h5py
import numpy as np
from function import activefunc
import matplotlib.pyplot as plt

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

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

print("train_set_x_orig:",train_set_x_orig.shape)
print("train_set_y:",train_set_y.shape)
print("test_set_x_orig:",test_set_x_orig.shape)
print("test_set_y:",test_set_y.shape)
print("classes:",classes.shape)

#矩阵转置
train_x=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_x=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
train_y=train_set_y.reshape(train_set_y.shape[1],1).T
test_y=test_set_y.reshape(test_set_y.shape[1],1).T
list_classes=classes.reshape(classes.shape[0],1).T

print("train_x:",train_x.shape)
print("train_y:",train_y.shape)
print("test_x:",test_x.shape)
print("test_y:",test_y.shape)
print("list_classes:",list_classes.shape)

#归一化
train_x=train_x/255
test_x=test_x/255

#w,b参数初始化
def init_w_b(dim_x,dim_y):
    W=0.01*np.random.randn(dim_x,dim_y)
    b=0
    return W,b
W,b=init_w_b(train_x.shape[0],1)

def logic_train(train_x,train_y,W,b,learn_rate,iter_times):
    #计算样本数：
    train_m=train_x.shape[1]
    cost=np.zeros(iter_times)
    for i in range(iter_times):
        #计算Z
        Z=np.dot(W.T,train_x)+b
        #计算A
        A=activefunc.sigmoid(Z)
        #计算代价函数
        cost[i]=-1.0/train_m*np.sum(train_y*np.log(A)+(1-train_y)*np.log(1-A))
        # print("Z:",Z.shape)
        # print("A:",A.shape)
        # print("cost:",cost[i])

        #计算反向传播
        dZ=A-train_y
        # print("dZ:",dZ.shape)

        dW=1.0/train_m*np.dot(train_x,dZ.T)
        # print("dW:",dW.shape)
        db=1.0/train_m*np.sum(dZ)
        # print("db:",db)

        #更新参数权值
        W=W-learn_rate*dW
        b=b-learn_rate*db
        # print("W:",W.shape)
        # print("b:",b)
    return W,b,cost

W,b,cost=logic_train(train_x,train_y,W,b,learn_rate=0.001,iter_times=2000)

def predict_y(test_x,W,b):
    #测试用例样本数
    test_m=test_x.shape[1]
    prediction_Y=np.zeros((1,test_m))
    #计算
    test_Z=np.dot(W.T,test_x)+b
    #计算预测，大于0.5输出1，少于0.5输出0
    test_Y=activefunc.sigmoid(test_Z)

    for i in range(test_Y.shape[1]):
        if test_Y[0][i]>0.5:
            prediction_Y[0][i]=1
        else:
            prediction_Y[0][i]=0

    return prediction_Y

train_Y_hat=predict_y(train_x,W,b)
print("train_Y_hat:",train_Y_hat.shape)
print("train_y:",train_y.shape)
print("train accuracy: {} %".format(100 - np.mean(np.abs(train_Y_hat-train_y)) * 100))

test_Y_hat=predict_y(test_x,W,b)
print("prediction_Y:",test_Y_hat.shape)
print("test_y:",test_y.shape)
print("test accuracy: {} %".format(100 - np.mean(np.abs(test_Y_hat-test_y)) * 100))

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

my_image_y=predict_y(my_image_x,W,b)
print(my_image_y)


# plt.plot(cost)
# plt.xlabel('iter times')
# plt.ylabel('cost')
plt.show()

