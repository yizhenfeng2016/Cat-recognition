#coding:utf-8
__author__ = 'Administrator'

import numpy as np

def normalizeRows(x):
    x_norms=np.linalg.norm(x,axis=1,keepdims=True) #计算每一行的模，x^2=x0^2+x1^2+x2^2+....
    return x/x_norms

def normalizeCols(x):
    pass

def softmax(x):
    #多项式回归
    pass