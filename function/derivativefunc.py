#coding:utf-8
__author__ = 'Administrator'

import numpy as np
from function import activefunc

def sigmoid_derivative(x):
    s=activefunc.sigmoid(x)
    ds=s*(1-s)
    return ds

def tanh_derivative(x):
    s=activefunc.tanh(x)
    ds=1-s**2
    return ds