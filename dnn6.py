import numpy as np
import matplotlib.pyplot as plt
import math
from dnn5 import *


def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    使用梯度下降更新参数
    
    输入:
    parameters -- 待更新的参数字典:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- 待更新的梯度字典:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    learning_rate -- 学习率，标量.
    
    输出:
    parameters -- 已更新的参数字典 
    """

    L = len(parameters) // 2 # 网络层数

    # 每个参数的更新规则
    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads['dW' + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads['db' + str(l+1)]
        
    return parameters


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    从(X, Y)中创造一个包含无序mini_batches的列表
    
    输入:
    X -- 输入数据，大小为(输入样本大小, 样本个数)
    Y -- 真实标签向量(1猫/ 0非猫), 形状为(1, 样本个数)
    mini_batch_size -- mini-batches的大小, 整数
    
    输出:
    mini_batches -- 同步的包含(mini_batch_X, mini_batch_Y)的列表
    """
    
    m = X.shape[1]                  # 训练集大小
    mini_batches = []
        
    # 第一步:将(X, Y)打乱
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    # 第二布: 划分 (shuffled_X, shuffled_Y). 不包含最后一部分
    num_complete_minibatches = math.floor(m/mini_batch_size) # 小块的个数
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k * mini_batch_size: (k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k * mini_batch_size: (k+1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # 处理最后一部分 (最后一个小块的大小 < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, mini_batch_size * num_complete_minibatches : m]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * num_complete_minibatches : m]
        # mini_batch_X = shuffled_X[:,k * mini_batch_size: (k) * mini_batch_size + m - num_complete_minibatches * mini_batch_size]
        # mini_batch_Y = shuffled_Y[:,k * mini_batch_size: (k) * mini_batch_size + m - num_complete_minibatches * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches  


def initialize_velocity(parameters):
    """
    用下面的参数将速度初始化为字典:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: 和梯度/参数 大小一样的0 numpy数组
    输入:
    parameters -- 包含参数的字典
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    
    输出:
    v -- 包含目前速度的字典
                    v['dW' + str(l)] = dWl 的速度
                    v['db' + str(l)] = dbl 的速度
    """
    
    L = len(parameters) // 2 # 神经网络的层数
    v = {}
    
    # I初始化速度
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros((parameters['W' + str(l+1)].shape))
        v["db" + str(l+1)] = np.zeros((parameters['b' + str(l+1)].shape))
        
    return v   


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    用Momentum方法更新参数
    
    输入:
    parameters -- 参数字典:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- 参数梯度字典:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- 当前速度字典:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- momentum方法的超参数，标量
    learning_rate -- 学习率，标量
    
    输出:
    parameters -- 更新后的参数字典 
    v -- 更新后速度的字典
    """

    L = len(parameters) // 2 # 神经网络的层数
    
    # 对每个参数使用Momentum方法进行更新
    for l in range(L):
        
        # 计算速度
        v["dW" + str(l+1)] = beta * v["dW" + str(l+1)] + (1-beta) *  grads['dW' + str(l+1)]
        v["db" + str(l+1)] = beta * v["db" + str(l+1)] + (1-beta) *  grads['db' + str(l+1)]
        # 更新参数
        parameters["W" + str(l+1)] -= learning_rate * v["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * v["db" + str(l+1)]
        
    return parameters, v


def initialize_adam(parameters) :
    """
    用以下参数更新v和s:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: 和梯度/参数 大小一样的0 numpy数组
    
    输入:
    parameters -- 参数字典
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    
    输出: 
    v -- 包含梯度指数加权平均值的python字典
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- 将包含指数加权平均平方梯度的python字典，。
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """
    
    L = len(parameters) // 2 # 神经网络的层数
    v = {}
    s = {}
    
    # 初始化 v, s. 输入: "parameters". 输出: "v, s".
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros((parameters["W" + str(l+1)].shape))
        v["db" + str(l+1)] = np.zeros((parameters["b" + str(l+1)].shape))
        s["dW" + str(l+1)] = np.zeros((parameters["W" + str(l+1)].shape))
        s["db" + str(l+1)] = np.zeros((parameters["b" + str(l+1)].shape))
    
    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    用Adam方法更新参数
    
    输入:
    parameters -- 参数字典:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- 梯度字典:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam方法的变量, 第一个梯度的移动平均, python字典
    s -- Adam方法的变量, 平方梯度的移动平均, python字典
    learning_rate -- 学习率, 标量
    beta1 -- 第一个矩估计的指数衰减超参数 
    beta2 -- 二阶矩的指数衰减超参数估计
    epsilon -- 在Adam更新中防止除0的超参数

    输出:
    parameters -- 包含更新参数的python字典
    v -- Adam方法的变量, 第一个梯度的移动平均, python字典
    s -- Adam方法的变量, 平方梯度的移动平均, python字典
    """
    
    L = len(parameters) // 2                 # 神经网络的层数
    v_corrected = {}                         # 初始化第一个矩估计，python字典
    s_corrected = {}                         # 初始化二阶矩估计，python字典
    
    # 对所有参数执行Adam更新
    for l in range(L):
        # 梯度的移动平均 输入: "v, grads, beta1". 输出: "v".
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1-beta1) * grads['dW' + str(l+1)]
        v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1-beta1) * grads['db' + str(l+1)]

        # 计算经校正的首弯矩估计值。 输入: "v, beta1, t". 输出: "v_corrected".
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - np.power(beta1,t))
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1 - np.power(beta1,t))

        # 梯度的平方的移动平均  输入: "s, grads, beta2". 输出: "s".
        s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1-beta2) * np.square(grads['dW' + str(l+1)])
        s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1-beta2) * np.square(grads['db' + str(l+1)])

        # 计算偏差校正的第二原始矩估计  输入: "s, beta2, t". 输出: "s_corrected".
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] /  (1 - np.power(beta2,t))
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] /  (1 - np.power(beta2,t))

        # 更新参数  输入: "parameters, learning_rate, v_corrected, s_corrected, epsilon". 输出: "parameters".
        parameters["W" + str(l+1)] -= learning_rate * v_corrected["dW" + str(l+1)] / np.sqrt(s_corrected["dW" + str(l+1)] + epsilon)
        parameters["b" + str(l+1)] -= learning_rate * v_corrected["db" + str(l+1)] / np.sqrt(s_corrected["db" + str(l+1)] + epsilon)

    return parameters, v, s


def model(X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 1000, print_cost = True):
    """
    可以在不同的优化模式下运行L层神经网络模型。
    
    输入:
    X -- 输入数据, 形状为(样本大小, 样本个数)
    Y -- 真实标签向量(1 猫 / 0 非猫), 形状为 (1, 样本个数)
    layers_dims -- 包含每层神经元数的字典
    learning_rate -- 学习率，标量
    mini_batch_size -- 小块的大小
    beta -- Momentum 超参数
    beta1 -- 指数衰减超参数的过去梯度估计
    beta2 -- 指数衰减超参数为过去的平方梯度估计 
    epsilon -- 在Adam更新中防止除0的超参数
    num_epochs -- 迭代的次数
    print_cost -- 是否打印代价

    输出:
    parameters -- 更新后的参数字典 
    """

    L = len(layers_dims)             # 神经网络的层数
    costs = []                       # 存代价
    t = 0                            # 初始化Adam更新所需的计数器
    seed = 10                         # 随机种子
    
    # 初始化参数
    parameters = initialize_parameters(layers_dims)

    # 初始换优化器
    if optimizer == "gd":
        pass # 梯度下降不需要初始化
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
    
    # 优化循环
    for i in range(num_epochs):
        
        # 定义随机的小批。在每个迭代之后，增加种子来对数据集进行不同的重新洗牌
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:

            # 选择一个小块
            (minibatch_X, minibatch_Y) = minibatch

            # 前向传播
            al, caches = forward_propagation(minibatch_X, parameters)

            # 计算代价
            cost = compute_cost(al, minibatch_Y)

            # 后向传播
            grads = backward_propagation(al, minibatch_Y, caches)

            # 更新参数
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1 # Adam 计数器
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2,  epsilon)
        
        # 打印代价
        if print_cost and i % 20 == 0:
            print ("Cost after epoch %i: %f" %(i, cost))
        if print_cost and i % 10 == 0:
            costs.append(cost)
                
    # 绘制代价
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 10)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters