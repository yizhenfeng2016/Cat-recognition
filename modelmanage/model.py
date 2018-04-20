#coding:utf-8
__author__ = 'Administrator'
import numpy as np

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    A = 1/(1+np.exp(-Z))
    return A

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    return dZ

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    return A

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

def test_relu_backward():
    dA=np.array([-1,1,-1])
    Z=np.array([-1,1,-1])
    print(relu_backward(dA,Z))
# test_relu_backward()
#
def params_init_model(input_dims,layer_num,cell_list,init_method="Normal",seed=0):
    '''
    :param input_dims: 输入数据X的维度
    :param layer_num: 有多少层
    :param cell_list: 每一层有多少个神经元，并作成列表传入
    :return:params字典
    '''
    np.random.seed(seed)
    params={}
    l_prew_cell=input_dims
    for i in range(1,layer_num+1):
        l_cell=cell_list[i-1]
        if init_method=="Xavier":
            w=np.random.randn(l_cell,l_prew_cell)*np.sqrt(1/l_prew_cell)
        elif init_method=="He":
            w=np.random.randn(l_cell,l_prew_cell)*np.sqrt(2/l_prew_cell)
        else:
            w=np.random.randn(l_cell,l_prew_cell)*0.01
        b=np.zeros((l_cell,1))
        l_prew_cell=l_cell
        params["W"+str(i)]=w
        params["b"+str(i)]=b
    return params

def test_params_init_model():
    p=params_init_model(5,3,[3,2,1])
    print(p["W2"])


def l_layer_forward_model(A_prew,W,b,acitvation=""):
   '''

   :param A_prew: 输入数据
   :param W: 参数W
   :param b: 偏置b
   :param acitvation: 激活函数
   :return:A,Z
   '''
   Z=np.dot(W,A_prew)+b
   assert(Z.shape==(W.shape[0],A_prew.shape[1]))
   if acitvation=="sigmoid":
       A=sigmoid(Z)
   elif acitvation=="relu":
       A=relu(Z)
   else:
       A=Z
   return A,Z

def test_layer_forward_model():
    a=np.random.randn(5,3)
    w=np.random.randn(2,5)
    b=1
    A,Z=l_layer_forward_model(a,w,b,'relu')
    print(A)
    print(Z)

# test_layer_forward_model()

def l_layer_forward_with_dropout_model(A_prew,W,b,keep_prob=1,acitvation="",seed=0):
   '''

   :param A_prew: 输入数据
   :param W: 参数W
   :param b: 偏置b
   :param acitvation: 激活函数
   :return:A,Z
   '''
   np.random.seed(seed)
   mask = np.random.binomial(1,keep_prob, A_prew.shape)
   A_prew=np.multiply(A_prew,mask)
   A_prew=A_prew/keep_prob

   Z=np.dot(W,A_prew)+b
   assert(Z.shape==(W.shape[0],A_prew.shape[1]))
   if acitvation=="sigmoid":
       A=sigmoid(Z)
   elif acitvation=="relu":
       A=relu(Z)
   else:
       A=Z
   return A,Z,mask

def cost_model(y_hat,train_y):
    '''

    :param y_hat: 训练的最终输出
    :param train_y:数据标签
    :param train_m:数据样本
    :return:J,dJ
    '''
    train_m=train_y.shape[1]
    J=-1.0/train_m*np.sum(train_y*np.log(y_hat)+(1-train_y)*np.log(1-y_hat))
    dJ=-(np.divide(train_y,y_hat)-np.divide(1-train_y,1-y_hat))
    # -train_y/y_hat+(1-train_y)/(1-y_hat)
    # - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    return J,dJ

def test_cost_model():
    y_hat=np.abs(0.01*np.random.randn(1,5))
    print(y_hat)
    train_y=np.abs(0.01*np.random.randn(1,5))
    print(train_y)
    J,dJ=cost_model(y_hat,train_y)
    print(J)
    print(dJ)
# test_cost_model()

def cost_with_L2_model(y_hat,train_y,params,lambd):
    '''
    L2正则--减少过拟合
    :param y_hat: 训练的最终输出
    :param train_y:数据标签
    :param params:参数
    :param lambd:参数
    :return:J,dJ
    '''
    L=len(params)//2
    W_sum=0
    for l in range(1,L+1):
        W_sum=W_sum+np.sum(np.square(params["W"+str(l)]))
    train_m=train_y.shape[1]
    J=-1.0/train_m*np.sum(train_y*np.log(y_hat)+(1-train_y)*np.log(1-y_hat))
    L2_cost=1.0/train_m*lambd/2*W_sum
    J=J+L2_cost
    dJ=-(np.divide(train_y,y_hat)-np.divide(1-train_y,1-y_hat))
    # -train_y/y_hat+(1-train_y)/(1-y_hat)
    # - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    return J,dJ

def l_layer_backward_model(A0,W1,Z1,dA,acitvation=""):
    '''

    :param A0: 该层的输入
    :param Z1: 线性计算得出的中间值
    :param dA: 后一层的求得的导数
    :param acitvation:激活函数
    :return:dW，db
    '''
    train_m=A0.shape[1]
    if acitvation=="sigmoid":
        dZ=sigmoid_backward(dA,Z1)
    elif acitvation=="relu":
        dZ=relu_backward(dA,Z1)
    else:
        dZ=dA
    # print("dZ:",dZ.shape)
    assert (A0.shape[1]==dZ.shape[1])
    dW=1.0/train_m*np.dot(dZ,A0.T)
    db=1.0/train_m*np.sum(dZ,axis=1, keepdims=True)
    dA_prew=np.dot(W1.T,dZ)
    return dA_prew,dW,db

def l_layer_backward_with_L2_model(A0,W1,Z1,dA,lambd,acitvation=""):
    '''
    L2正则--减少过拟合
    :param A0: 该层的输入
    :param Z1: 线性计算得出的中间值
    :param dA: 后一层的求得的导数
    :param acitvation:激活函数
    :return:dW，db
    '''
    train_m=A0.shape[1]
    if acitvation=="sigmoid":
        dZ=sigmoid_backward(dA,Z1)
    elif acitvation=="relu":
        dZ=relu_backward(dA,Z1)
    else:
        dZ=dA
    # print("dZ:",dZ.shape)
    assert (A0.shape[1]==dZ.shape[1])
    dW=1.0/train_m*np.dot(dZ,A0.T)+lambd/train_m*W1
    db=1.0/train_m*np.sum(dZ,axis=1, keepdims=True)
    dA_prew=np.dot(W1.T,dZ)
    return dA_prew,dW,db

def test_layer_backward_model():
    y_hat=np.abs(0.01*np.random.randn(1,3))
    train_y=np.abs(0.01*np.random.randn(1,3))
    A0=0.01*np.random.randn(5,3)
    Z1=0.01*np.random.randn(1,3)
    J,dJ=cost_model(y_hat,train_y)
    print("dJ:",dJ.shape)
    dw,db=l_layer_backward_model(A0,Z1,dJ,acitvation="sigmoid")
    print(dw)
    print(db)
# test_layer_backward_model()

def update_params(params,grad,learning_rate=0.01):
    '''

    :param params:
    :param grad:
    :param learning_rate:
    :return:修正后的params
    '''
    L=len(params)//2
    for l in range(1,L):
        params["W"+str(l)]=params["W"+str(l)]-learning_rate*grad["dW"+str(l)]
        params["b"+str(l)]=params["b"+str(l)]-learning_rate*grad["db"+str(l)]
    return params

def l_layer_backward_with_dropout_model(A0,mask,W1,Z1,dA,keep_prob=1,acitvation=""):
    '''
    dropout--减少过拟合
    :param A0: 该层的输入
    :param Z1: 线性计算得出的中间值
    :param dA: 后一层的求得的导数
    :param acitvation:激活函数
    :return:dW，db
    '''
    train_m=A0.shape[1]
    if acitvation=="sigmoid":
        dZ=sigmoid_backward(dA,Z1)
    elif acitvation=="relu":
        dZ=relu_backward(dA,Z1)
    else:
        dZ=dA
    # print("dZ:",dZ.shape)
    assert (A0.shape[1]==dZ.shape[1])
    dW=1.0/train_m*np.dot(dZ,A0.T)
    db=1.0/train_m*np.sum(dZ,axis=1, keepdims=True)
    dA_prew=np.dot(W1.T,dZ)

    dA_prew=np.multiply(dA_prew,mask)
    dA_prew=dA_prew/keep_prob

    return dA_prew,dW,db


def mini_batch(X,Y,mini_batch_size=64,seed=0):
    import math
    '''
    :param X: train_x
    :param Y: train_y
    :param mini_batch_size: 小批量样本数
    :param seed:随机种子
    :return:（x,y）每一块的列表
    '''
    np.random.seed(seed)
    train_m=X.shape[1]
    shuffle=np.random.permutation(train_m)
    shuffle_x=X[:,shuffle]
    shuffle_y=Y[:,shuffle]
    batch_num=math.floor(train_m/mini_batch_size) #多少块
    batches_list=[]
    # if batch_num:
    #     for k in range(batch_num):
    #         batch_x=shuffle_x[:,k*mini_batch_size:(k+1)*mini_batch_size]
    #         batch_y=shuffle_y[:,k*mini_batch_size:(k+1)*mini_batch_size]
    #         batch=(batch_x,batch_y)
    #         batches_list.append(batch)
    #     if batch_num*mini_batch_size<train_m:
    #         batch_x=shuffle_x[:,batch_num*mini_batch_size:train_m]
    #         batch_y=shuffle_y[:,batch_num*mini_batch_size:train_m]
    #         batch=(batch_x,batch_y)
    #         batches_list.append(batch)
    # else:
    #     batch_x=shuffle_x[:,0:train_m]
    #     batch_y=shuffle_y[:,0:train_m]
    #     batch=(batch_x,batch_y)
    #     batches_list.append(batch)

    for k in range(batch_num):
        batch_x=shuffle_x[:,k*mini_batch_size:(k+1)*mini_batch_size]
        batch_y=shuffle_y[:,k*mini_batch_size:(k+1)*mini_batch_size]
        batch=(batch_x,batch_y)
        batches_list.append(batch)
    if train_m%mini_batch_size!=0:
        batch_x=shuffle_x[:,batch_num*mini_batch_size:train_m]
        batch_y=shuffle_y[:,batch_num*mini_batch_size:train_m]
        batch=(batch_x,batch_y)
        batches_list.append(batch)

    return batches_list

def test_mini_batch():
    x=np.array([[1,2,3,4,5,6,7],[1,2,3,4,5,6,7]])
    y=np.array([[1,3,5,7,9,11,13],[1,3,5,7,9,11,13]])
    li=mini_batch(x,y,3)
    for l in li:
        x,y=l
        print("yici")
        print(x)
        print(y)

# test_mini_batch()

#Adam优化
def init_adam(params):
    '''

    :param params: 参数初始化
    :return:
    v--weighted average of the gradient
    s--weighted average of the squared gradient
    '''
    L=len(params)//2 #除法取整数
    v={}
    s={}
    for l in range(1,L+1):
        v["dW"+str(l)]=np.zeros(params["W"+str(l)].shape)
        v["db"+str(l)]=np.zeros(params["b"+str(l)].shape)
        s["dW"+str(l)]=np.zeros(params["W"+str(l)].shape)
        s["db"+str(l)]=np.zeros(params["b"+str(l)].shape)
    return v,s

def update_params_with_adam(params,grad,learning_rate,v,s,t=2,beta1=0.9,beta2=0.999,epsilon=1e-8):
    '''
     adam--能减少梯度下降时的波动
    :param params: w,b参数
    :param grad: dw,db参数
    :param v:
    :param s:
    :param t: 修正参数
    :param beta1:
    :param beta2:
    :param learning_rate:
    :param epsilon:增加项，防止分母为0
    :return:修改过的params，以及更新的v,s,用于下一次迭代
    '''
    L=len(params)//2
    v_correct={}
    s_correct={}
    for l in range(1,L+1):
        v["dW"+str(l)]=beta1*v["dW"+str(l)]+(1-beta1)*grad["dW"+str(l)]
        v_correct["dW"+str(l)]=v["dW"+str(l)]/(1-beta1**t)
        s["dW"+str(l)]=beta2*s["dW"+str(l)]+(1-beta2)*grad["dW"+str(l)]**2
        s_correct["dW"+str(l)]=s["dW"+str(l)]/(1-beta2**t)
        params["W"+str(l)]=params["W"+str(l)]-learning_rate*v_correct["dW"+str(l)]/(np.sqrt(s_correct["dW"+str(l)])+epsilon)
        
        v["db"+str(l)]=beta1*v["db"+str(l)]+(1-beta1)*grad["db"+str(l)]
        v_correct["db"+str(l)]=v["db"+str(l)]/(1-beta1**t)
        s["db"+str(l)]=beta2*s["db"+str(l)]+(1-beta2)*grad["db"+str(l)]**2
        s_correct["db"+str(l)]=s["db"+str(l)]/(1-beta2**t)
        params["b"+str(l)]=params["b"+str(l)]-learning_rate*v_correct["db"+str(l)]/(np.sqrt(s_correct["db"+str(l)])+epsilon)

    return params,v,s

