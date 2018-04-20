#coding:utf-8
__author__ = 'Administrator'

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

def load_dataset():
    """
    # 加载数据集
    """
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")  # 读取H5文件

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

    #矩阵
    train_x=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1)
    test_x=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1)
    train_y=train_set_y_orig.reshape(train_set_y_orig.shape[1],1)
    test_y=test_set_y_orig.reshape(test_set_y_orig.shape[1],1)
    list_classes=classes.reshape(classes.shape[0],1)
    return train_x, train_y, test_x, test_y, list_classes


if __name__=="__main__":
    #数据集
    train_x, train_y, test_x, test_y, list_classes=load_dataset()
    #归一化
    train_x=train_x/255
    test_x=test_x/255
    print('x shape:{}'.format(np.shape(train_x)))
    print('y shape:{}'.format(np.shape(train_y)))

    #定义输入输出的变量，没有值，只“占位”
    x=tf.placeholder(dtype=tf.float32,shape=[None,12288])
    y=tf.placeholder(dtype=tf.float32,shape=[None,1])
    y_hat=tf.placeholder(dtype=tf.float32,shape=[None,1])
    y_p=tf.placeholder(dtype=tf.float32,shape=[None,1])

    ###模型###
    #第一层
    dense1=tf.layers.dense(inputs=x,units=20,activation=tf.nn.relu)
    #第二层
    dense2=tf.layers.dense(inputs=dense1,units=7,activation=tf.nn.relu)
    #第三层
    dense3=tf.layers.dense(inputs=dense2,units=5,activation=tf.nn.relu)
    #输出层
    output=tf.layers.dense(inputs=dense3,units=1,activation=tf.nn.sigmoid)

    #用交叉熵计算损失
    loss=tf.reduce_mean(-tf.reduce_sum(y*tf.log(output)+(1-y)*tf.log(1-output))) #y*tf.log(dense4)+(1-y)*tf.log(1-dense4)
    #使用梯度下降算法以0.0001的学习率最小化交叉墒
    train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

    #预测结果
    accuracy=100-tf.reduce_mean(tf.abs(y-y_hat))

    #初始化
    init=tf.global_variables_initializer()

    #保存数据
    save_dir='./model/'
    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(init)#启动初始化
        for i in range(501):
            step,cost,y_p=sess.run([train_step,loss,output],feed_dict={x:train_x,y:train_y})
            if i%50==0:
                # print('output shape:{}'.format(np.shape(y_p)))
                print('Epoch {} rain_loss = {}'.format(i,cost))

        #概率大于0.5，则为1
        y_=np.ones((209,1))
        for i in range(y_p.shape[0]):
            if y_p[i][0]>0.5:
                y_[i][0]=1
            else:
                y_[i][0]=0

        # #在session中启动accuracy，输入是训练集
        train_accuracy=sess.run(accuracy, feed_dict={y:train_y, y_hat:y_})
        print('train_accuracy:{}'.format(train_accuracy))

        #把测试集输入到训练好的模型，计算出概率
        y_p=sess.run(output,feed_dict={x:test_x})
        # print('output shape:{}'.format(np.shape(y_p)))
        #概率大于0.5，则为1
        y_=np.ones((np.shape(test_x)[0],1))
        for i in range(y_p.shape[0]):
            if y_p[i][0]>0.5:
                y_[i][0]=1
            else:
                y_[i][0]=0

        # #在session中启动accuracy，输入是测试集
        test_accuracy=sess.run(accuracy, feed_dict={y:test_y, y_hat:y_})
        print('test_accuracy:{}'.format(test_accuracy))

        saver.save(sess,save_dir)
        print("Training finished and save model")
