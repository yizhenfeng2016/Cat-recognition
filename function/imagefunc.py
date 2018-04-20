#coding:utf-8
__author__ = 'Administrator'

import numpy as np

def image2vector(image):
    vector=image.reshape(image.shape[0]*image.shape[1]*image.shape[2],1)
    return vector