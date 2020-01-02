import numpy as np
import matplotlib.pyplot as plt
import h5py


def softmax(Z):
    """
    实现softmax激活函数
    
    输入:
    Z -- 任意形状的numpy数组
    
    返回参数:
    A -- softmax(Z)的输出, 形状和Z一样
    cache -- 也返回Z, 在计算反向传播时有用
    """

    Z_exp = np.exp(Z)
    partition = np.sum(Z_exp,axis=0,keepdims=True)
    A = Z_exp / partition
    cache = Z 

    return A, cache


def relu(Z):
    """
    实现RELU函数.

    输入:
    Z -- 线性单元的输出

    返回参数:
    A -- 激活后参数, 形状和Z一样
    cache -- 一个存储反向传播所需参数的字典
    """

    A = np.maximum(0, Z)

    assert (A.shape == Z.shape)

    cache = Z
    return A, cache


def relu_backward(dA, cache):
    """
    实现单个的RELU神经元的后向传播

    输入:
    dA -- 上一层激活函数的导数
    cache -- “Z”，一个存储反向传播所需参数的字典

    返回参数:
    dZ -- 对Z的导数
    """

    Z = cache
    dZ = np.array(dA, copy=True)  # 转化为能用的数据

    # 当 z <= 0, 使 dz 也为 0
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def softmax_backward(dA, cache):
    """
    实现单个的softmax神经元的后向传播

    输入:
    dA -- 上一层激活函数的导数
    cache -- “Z”，一个存储反向传播所需参数的字典

    返回参数:
    dZ -- 对Z的导数
    """

    Z = cache
    Z_exp = np.exp(Z)
    partition = np.sum(Z_exp,axis=0,keepdims=True)
    s = Z_exp / partition  
    dZ = dA * s + s

    assert (dZ.shape == Z.shape)

    return dZ


# def load_data():
#     train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
#     train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # 训练集
#     train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # 训练集预先设好的标签

#     test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
#     test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # 验证集特征
#     test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # 验证集预先设好的标签

#     classes = np.array(test_dataset["list_classes"][:])  # 结果列表

#     train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
#     test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

#     return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def initialize_parameters(layer_dims):
    """
    参数初始化

    输入:
    layer_dims -- 包含每层网络维数（神经元个数）的字典
    
    返回参数:
    parameters -- 一个包含参数 "W1", "b1", ..., "WL", "bL"的字典:
                    Wl -- 形状为(layer_dims[l], layer_dims[l-1])的权重矩阵
                    bl -- 形状为(layer_dims[l], 1)的偏差向量
    """

    # np.random.seed(1)
    parameters = {}
    L = len(layer_dims)  # 网络的层数

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l-1])  # He initialization
        # parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])  # *0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    """
    实现一个线性层的前向传播

    输入:
    A -- 上一层的激活单元（或输入数据），形状为: (上一层神经元个数, 样本个数)
    W -- 权重矩阵: 有着(本层神经元个数, 上一层神经元个数)这样形状的numpy矩阵
    b -- 偏差向量, 有着(本层神经元个数, 1)这样形状的numpy矩阵

    返回参数:
    Z -- 激活函数的输入 
    cache -- 一个包含反向传播所需参数"A", "W" 和 "b" 的字典
    """

    Z = W.dot(A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    实现 LINEAR->ACTIVATION 层的前向传播

    输入:
    A_prev -- 上一层的激活单元（或输入数据），形状为: (上一层神经元个数, 样本个数)
    W -- 权重矩阵: 有着(本层神经元个数, 上一层神经元个数)这样形状的numpy矩阵
    b -- 偏差向量, 有着(本层神经元个数, 1)这样形状的numpy矩阵
    activation -- 用于这一层的激活开关, 存储着 "softmax" 或者 "relu" 字符串

    返回参数:
    A -- 激活函数的输出
    cache -- 一个包含反向传播所需参数 "linear_cache" 和 "activation_cache"的字典;
    """

    if activation == "softmax":
        # 输入: "A_prev, W, b". 输出: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)

    elif activation == "relu":
        # 输入: "A_prev, W, b". 输出: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def forward_propagation(X, parameters):
    """
    实现 [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID 层的前向传播，即实现整体的前向传播
    
    输入:
    X -- 输入数据, 形如(输入参数大小, 样本个数)
    parameters -- initialize_parameters()的输出
    
    返回参数:
    AL -- 最后一层的输出
    caches -- 列表里包含:
                使用激活开关"relu"驱动的linear_activation_forward()的输出(有(L-1)个, 下标从0到L-2)
                使用激活开关"sigmoid"驱动的linear_activation_forward()的输出(一个, 下标为L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2  # 网络层数

    # 实现 [LINEAR -> RELU]*(L-1). 添加 "cache" 进入 "caches" 列表
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                             activation="relu")
        caches.append(cache)

    # 实现 LINEAR -> SOFTMAX. 添加 "cache" 进入 "caches" 列表
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="softmax")
    caches.append(cache)

    assert (AL.shape == (10, X.shape[1]))

    return AL, caches


def compute_cost(AL, Y):
    """
    实现代价函数

    输入:
    AL -- 经过训练产生的预测（标签）, 形状为(1, 样本个数)
    Y -- 真实标签向量 (例如: 如果不是猫为0, 是猫为1), 形状为(1, 样本个数)

    返回参数:
    cost -- 定义的代价
    """

    m = Y.shape[1]

    # 计算产生自AL和Y的代价
    # cost = (1. / m) * np.sum(-np.multiply(Y, np.log(AL)))
    # cost = (1. / m) * np.sum(-np.dot(Y, np.log(AL).T))
    cost = (1. / m) * np.sum(Y != np.argmax(AL, axis=0).reshape(1, m))

    cost = np.squeeze(cost)  # 使结果为想要的形状(e.g. 将 [[17]] 转化为 17).
    assert (cost.shape == ())

    return cost


def linear_backward(dZ, cache):
    """
    实现一层网络线性部分的反向传播 (某一层 即l层)

    输入:
    dZ -- 线性单元的导数 (属于当前层 即l层)
    cache -- 来自当前层前向传播的值为(A_prev, W, b)的元组 

    返回参数:
    dA_prev --激活单元的导数 (属于上一层 即l-1层), 和A_prev形状一样
    dW -- 关于W的导数 (当前层l), 和W形状一样
    db -- 关于b的导数 (当前层l), 和b形状一样
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    计算 LINEAR->ACTIVATION 层的导数
    
    输入:
    dA -- 本层，即l层激活单元的导数  
    cache -- 为计算反向传播存储的值为(linear_cache, activation_cache)的元组
    activation -- 用于本层的激活开关, 存储着"sigmoid" 或 "relu"的字符串
    
    返回参数:
    dA_prev -- 激活单元的导数 (属于上一层 即l-1层), 和A_prev形状一样
    dW -- 关于W的导数 (当前层l), 和W形状一样
    db -- 关于b的导数 (当前层l), 和b形状一样
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def backward_propagation(AL, Y, caches):
    """
    实现[LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID 的反向传播，即整体的反向传播
    
    输入:
    AL -- 前向传播的输出 (forward_propagation())
    Y -- 真实标签向量 (非猫0，猫1)
    caches -- 列表包含:
                使用激活开关"relu"驱动的linear_activation_forward()的输出(有(L-1)个, 下标从0到L-2)
                使用激活开关"sigmoid"驱动的linear_activation_forward()的输出(一个, 下标为L-1)
    
    返回参数:
    grads -- 含有导数的字典：
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches)  # 层数
    m = AL.shape[1]
    # Y = Y.reshape(AL.shape)  # 使Y和AL形状一样

    Y_temp = np.zeros(AL.shape)
    for i in range(AL.shape[1]):
        Y_temp[Y[0,i], i] = 1
    Y = Y_temp
    
    # 初始化反向传播
    dAL = - np.divide(Y, AL)

    # L层 (SIGMOID -> LINEAR) 导数. 输入: "AL, Y, caches". 输出: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,
                                                                                                  activation="softmax")

    for l in reversed(range(L - 1)):
        # l层: (RELU -> LINEAR) 导数.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache,
                                                                    activation="relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# def update_parameters(parameters, grads, learning_rate):
#     """
#     Update parameters using gradient descent
    
#     输入:
#     parameters -- python dictionary containing your parameters 
#     grads -- python dictionary containing your gradients, output of L_model_backward
    
#     返回参数:
#     parameters -- python dictionary containing your updated parameters 
#                   parameters["W" + str(l)] = ... 
#                   parameters["b" + str(l)] = ...
#     """

#     L = len(parameters) // 2  # number of layers in the neural network

#     # Update rule for each parameter. Use a for loop.
#     for l in range(L):
#         parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
#         parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

#     return parameters


def predict(X, y, parameters):
    """
    预测L层网络的结果
    
    输入:
    X -- 样本集
    parameters -- 训练参数
    
    返回参数:
    p -- 由样本集产生的预测（评估）
    """

    m = X.shape[1]
    n = len(parameters) // 2  # 网络层数
    p = np.zeros((1, m))
    num = 0

    # 前向传播
    probas, caches = forward_propagation(X, parameters)

    # 将输出转化为 0/1 
    p = (y == np.argmax(probas,axis=0))

    # 打印结果
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    print("Accuracy: " + str(np.sum(p) / m))

    return p


# def predict(X, y, parameters):
#     """
#     This function is used to predict the results of a  n-layer neural network.
    
#     输入:
#     X -- data set of examples you would like to label
#     parameters -- parameters of the trained model
    
#     返回参数:
#     p -- predictions for the given dataset X
#     """
    
#     m = X.shape[1]
#     p = np.zeros((1,m), dtype = np.int)
    
#     # Forward propagation
#     a3, caches = forward_propagation(X, parameters)
    
#     # convert probas to 0/1 predictions
#     for i in range(0, a3.shape[1]):
#         if a3[0,i] > 0.5:
#             p[0,i] = 1
#         else:
#             p[0,i] = 0

#     # print results

#     #print ("predictions: " + str(p[0,:]))
#     #print ("true labels: " + str(y[0,:]))
#     print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))
    
#     return p


# def print_mislabeled_images(classes, X, y, p):
#     """
#     绘制判断错误的图像
#     X -- 数据集
#     y -- 真实标签
#     p -- 预测
#     """
#     a = p + y
#     mislabeled_indices = np.asarray(np.where(a == 1))
#     plt.rcParams['figure.figsize'] = (40.0, 40.0)  # 设置默认图片大小
#     num_images = len(mislabeled_indices[0])
#     for i in range(num_images):
#         index = mislabeled_indices[1][i]

#         plt.subplot(2, num_images, i + 1)
#         plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
#         plt.axis('off')
#         plt.title(
#             "Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n Class: " + classes[y[0, index]].decode(
#                 "utf-8"))
