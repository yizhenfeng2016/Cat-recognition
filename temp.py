#coding:utf-8
__author__ = 'Administrator'

import numpy as np
import os

# d={"W1":1,"W3":[2,3,4,5,6],"D1":[345,44]}
# print(len(d))
# file,ext=os.path.splitext(os.path.basename(os.path.realpath(__file__)))
# print(file)
#
# x=np.random.randn(10,3)
# print(x)
# # shuffle=np.random.permutation(10)
# # print(shuffle)
# # l_x=x[:,shuffle]
# # print(l_x)
#
np.random.seed(0)
mask = np.random.binomial(1,0.5,10)
print(mask)
#
# x=np.multiply(x,mask)
# print(x)
#
# keep_prob=0.5
# d2=np.random.rand(1, 10)
# print(d2)
# d3 = d2 < keep_prob
# print(d3)
# # d=np.multiply(x,d3)
# # print(d)