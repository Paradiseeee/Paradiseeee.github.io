---
layout:     post
title:      "Neural Network and Deep Learning"
subtitle:   "Coursera 神经网络入门课程学习笔记"
date:       2021-07-30 12:00:00
author:     "Paradise"
header-style: text
tags:
    - 神经网络
    - 笔记
---


> [Coursera - Neural Network and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning#reviews)


# Introduction to Deep Learning

### Lesson 1: Welcome

- Neural Network and Deep Learning；
- Improving Deep Neural Networks: Hyperparameter tuning,Regularization and Optimization 以及一些高级优化算法，likeMomentum RMSProp 和 Adam 优化算法；
- Structuring your Machine Learning Project并分享系统学习的教材；
- Convolution Neural Networks;
- Natural Language Processing: Building sequence models.

### Lesson 2: What is a neural network?

- 简单线性回归可以看成是最简单的神经网络：input --> [a neuron] --> prediction
- ReLU函数：Rectified Linerar Unite，即max(0, y)，一部分为0，一部分为y。
- 更复杂的神经网络可以看作多元线性回归，同样以ReLU作为激活函数。与多元线性回归不同在于，神经网络具有隐藏神经元，构成全连接层（连接每一个输入特征）。

### Lesson 3: Supervised Learning with Neural Networks
- 什么是监督学习
- 计算机视觉中的数据是矩阵，常用卷积神经网络；语音或语言处理中的数据是序列数据，常用循环神经网络，RNN。
- 术语：Structured Data与Unstructured Data，前者是基于数据库的数据，每个特征有明确的定义和具体的数值；后者是指较原始的数据，如图像、音频、文本等。

### Lesson 4: Why is Deep Learning taking off?

- 电子化与信息化
- Data、Computation、Algorithm（Sigmoid-->ReLU）

### Omitted

<br>

# Neural Network Basics

### Lesson 1: Binary Classification

- 神经网络构造，前向传播与反相传播
- 以二分类为例介绍符号标注

### Lesson 2: Logistic Regression

- 模型公式：`y_hat = sigmoid(w^T * x + b)`，其中：`sigmoid(z) = 1 / [1 + e^(-z)]`

### Lesson 3: Logistic Regression Cost Function

- 损失函数示例：`L(y_hat, y) = 1/2 *(y_hat-y)^2`，但是这个函数非凸函数，使用梯度下降得不到全局最优解。
- logistic回归损失函数：`L(y_hat, y) = -[y*log(y_hat) + (1-y)*log(1-y_hat)]`
- logistic回归可以看作一个非常简单的神经网络。

### Lesson 4: Gradient Descent

- 使用梯度下降寻找最优w和b时，损失函数J(w,b)应该是凸函数（convex function），此时要做的就是寻找凸函数的最低点对应的参数。
- 梯度下降法：
    * 随机初始化w和b的值
    * 沿着最陡的下降方向走下坡路（下式直接用微分符号，数学上用偏微分符号）
        + 即迭代：`w = w - alpha * dJ(w,b)/dw`；`b = b - alpha * dJ(w,b)/db`
        + 其中alpha为学习率。
    * 到达“碗”的底部

### Lesson 5: Derivatives

- 数学的微分

### Lesson 6: More Derivative Examples

- 微分

### Lesson 7: Computation graph

- 计算图，前向传播（计算图的每个结点代表一个神经元，结点内容为神经元的特征组成）

### Lesson 8: Derivatives with a Computation Graph

- 计算图求导，反向传播（在神经元中反向求导）

### Lesson 9: Logistic Regression Gradient Descent

- 在计算图中计算偏导数，然后根据前面的公式迭代更新参数。

### Lesson 10: Gradient Descent on m Examples

- 上一节的情况为只有一个训练样本的情况，在含有m各训练样本时，分别使用每个样本对损失函数求偏导，然后用其均值进行梯度下降。（分别对每个参数）
- 上述过程涉及两个嵌套的循环：`for 每个参数:{for每个样本:{求偏导}}`；当样本较大或参数较多，不宜显式地使用循环，应用矢量化函数进行计算。

### Lesson 11: Vectorization

- 向量化使代码加速

### Lesson 12: More Vectorization Examples

- 更多向量化例子

### Lesson 13: Vectorizing Logistic Regression

- 使用python的广播机制进行向量化的逻辑斯蒂回归

### Lesson 14: Vectorizing Logistic Regression's Gradient Output

- 向量化计算梯度下降，主要运用numpy的dot函数，但仍需要使用for循环进行迭代。

### Lesson 15: Broadcasting in Python

- 参考python_crash_course

### Lesson 16: A note on python/numpy vectors

- 不要使用秩为1的数组，即shape=(n,)，改为shape=(n,1)，可以避免很多微妙的bug。

### Omitted

<br>

# Shallow Neural Networks

### Lesson 1: Neural Network Overview

- What's a  Neural Network? —— 参考logistic回归的模型，一个简单的神经网络；

### Lesson 2: Neural Network Representation

- 简单的分层：输入层、隐藏层、输出层（单个结点）；
- 计算层数时不计算输入层，参数所在的层数用带括号的上标表示；

### Lesson 3: Computing a Neural Network's Output

- 首先由 `w^T * x` 计算z，再由 `sigma(z)` 计算输出a，将参数和输入合并成矩阵进行向量化。
- `z^[l] = W[l]·x + b^[l]`; `a^[l] = sigma(z^[l]); l = 1, 2, 3, ...`

### Lesson 4: Vectorizing across multiple examples

- `for i in 1~m:`
    * `for l in 1~2:`
        + `z[1](i) = W[1]x(i) + b[1]`
        + `a[1](i) = sigma(z[1](i))`
- 再将上述操作向量化

### Lesson 5: Explanation for Vectorized Implementation

- 具体的向量操作

### Lesson 6: Activation functions

- 除了sigmoid函数作为激活函数，还可以选择其他非线性函数。
- `a = sigmoid(z) = 1 / [1 + e^(-z)]`
- tanh激活函数相当于sigmoid函数的中心化版本，效果优于sigmoid函数：
    * `a = tanh(z) = [e^z - e^(-z)] / [e^z + e^(-z)]`

### Lesson 7: Why do you need non-linear activation functions

- 如果不使用非线性激活函数（使用线性激活函数，或称恒等激活函数），所得到的隐藏层的输出呈线性相关关系，这样无论有多少层隐藏层都不起作用，相当于没有隐藏层的logistic回归模型。

### Lesson 8: Derivatives of activation functions

- 当在神经网络中使用反向传播时，需要计算激活函数的导数。
- sigmoid函数的导数：`g'(z) = g(z)(1 - g(z))`
- tanh函数的导数：`g'(z) = 1 - g(z)^2`

### Lesson 9: Gradient descent for Neural Networks

- 前向传播（下面括号一律代表上标）
    * `z[1] = w[1] * x + b[1]; A[1] = g[1](z[1])`
    * `z[2] = w[2] * A[1] + b[1]; A[2] = g[2](z[2]) = sigma(z[2])`
- 反向传播，对应于前向传播的逆过程
    * `dz[2] = A[2]·Y; Y = [y(1) y(2) ... y(m)]`
    * `dw[2] = 1/m * dz[1] * A[1]^T`
    * `db[2] = 1/m * np.sum(dz[2], axis=1, keepdims=True)`
    * `dz[1] = w[2]^T * dz[2] .* A[1]`
    * `dw[1] = 1/m * dz[1] * X^T`
    * `db[1] = 1/m * np.sum(dz[1], axis=1, keepdims=True)`

### Lesson 10: Backpropagation intuition (optional)

- 用计算图直观展示上述运算

### Lesson 11: Random Initialization

- 在logistic回归中可以直接将权重初始化为全零，但在神经网络中这样会导致隐藏层不同神经元的激活函数是完全一致的，同时导致进行反向传播时的导数也是完全一致的。并且在进行梯度下降迭代时，不同的隐藏单元实现的功能也是完全一致的，因此没有起到该有的作用。称为对称失效问题，对称指的是前向和反向的对称。
- 对于更深的神经网络，结果也是类似的。为了使不同的隐藏单元承担不同的计算，可以进行随机初始化。

### Lesson 12: Ian Goodfellow interview

- interesting：波兹曼机


<br>

# Deep Neural Networks

### Lesson 1: Deep L-layer neural network

- 零隐藏层：Logistic回归；一及以上：神经网络

### Lesson 2: Forward Propagation in a  Deep Netwrok

- 介绍符号标注和之前讲过的前向传播

### Lesson 3: Gettiing your matrix dimensions right

- 略

### Lesson 4: Why deep representations?

- 略

### Lesson 5: Building blocks of deep neural netwworks

- 略

### Lesson 6: Forward and Backward Propagation

- 略

### Lesson 7: Parameters vs Hyperparameters

- 最优的超参数不是一成不变的，如果研究的问题历时较长，建议定时进行参数调优。

### Lesson 8: What does this have to do with the brain?

- 与人脑的机制并没有多少本质上的关联！只是一个类比。
