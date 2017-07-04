# -*- coding:utf-8 -*-
# logistic回归

from numpy import *
import matplotlib.pyplot as plt


def loadDataSet():
    dataMat = []
    labelMat = []
    rf = open('testSet.txt')
    for line in rf.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    # 移动步长
    alpha = 0.001
    # 迭代次数
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    # 返回训练好的回归系数
    return weights


# 随机梯度上升算法
def stocGradAscent0(dataMatIn, classLabels):
    m, n = shape(dataMatIn)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        # 此处sum只是把计算得到的一个元素的向量转换为数字
        h = sigmoid(sum(dataMatIn[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatIn[i]
    return weights


# 改进的随机梯度上升算法
def stocGradAscent1(dataMatIn, classLabels, numIter=150):
    m, n = shape(dataMatIn)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatIn[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatIn[randIndex]
            del (dataIndex[randIndex])
    return weights


# 画出数据集和logistic回归最佳拟合直线
def plotBestFit():
    dataMat, labelMat = loadDataSet()
    # weights = gradAscent(dataMat, labelMat).getA()
    # weights = stocGradAscent0(array(dataMat), labelMat)
    weights = stocGradAscent1(array(dataMat), labelMat, 500)
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    # sigmoid函数中1/(1+e[-z次方]),z为0时,是函数二分的边界点,
    # 所以此处解函数 0=w0*x0+w1*x1+w2*x2,
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]

    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


plotBestFit()
