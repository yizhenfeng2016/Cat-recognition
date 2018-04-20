#coding:utf-8
__author__ = 'Administrator'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    #定义输入输出的变量，没有值，只“占位”
    x=tf.placeholder(dtype=tf.float32,shape=[None,12288])

    ###模型###
    #第一层
    dense1=tf.layers.dense(inputs=x,units=20,activation=tf.nn.relu)
    #第二层
    dense2=tf.layers.dense(inputs=dense1,units=7,activation=tf.nn.relu)
    #第三层
    dense3=tf.layers.dense(inputs=dense2,units=5,activation=tf.nn.relu)
    #输出层
    output=tf.layers.dense(inputs=dense3,units=1,activation=tf.nn.sigmoid)

    #加载数据
    save_dir='./model/'
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_dir+'.meta')   # 载入图结构，保存在.meta文件中
        saver.restore(sess,save_dir)    # 载入参数，参数保存在两个文件中，不过restore会自己寻找

        #实际图片进行测试
        from scipy import ndimage,misc
        from function import imagefunc
        import os

        image_dir="images/" #图片目录
        cat_count=0
        total=0
        for file in os.listdir(image_dir):
            total=total+1
            frame=os.path.join(image_dir,file)
            my_image_x=np.array(ndimage.imread(frame,flatten=False))
            my_image_x=misc.imresize(my_image_x,size=(64,64))
            # plt.imshow(my_image_x)
            my_image_x=my_image_x/255
            my_image_x=imagefunc.image2vector(my_image_x).T
            # print(my_image_x.shape)
            y_p=sess.run(output,feed_dict={x:my_image_x})
            if y_p[0]>0.5:
                cat_count=cat_count+1
                print('picture path:{} is cat, possibility is {}'.format(frame,y_p[0]))
            else:
                print('picture path:{} is not cat, possibility is {}'.format(frame,y_p[0]))
            # plt.show()
        print('real accuracy is {0:.4f}%'.format(cat_count/total))