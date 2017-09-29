# -*- coding:utf-8 -*-
# k临近算法Demo

from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

'''
inX:待分类元素
dataSet:样本集合
labels:标签集合
kk:取距离最近的样本个数
'''


def classify0(inX, dataSet, labels, kk):
    dataSetSize = dataSet.shape[0]
    # tile用于构造一个和dataSet相同阵列的集合,集合中的每一个元素都是inX,与dataSet做减法,相当于每一个元素都做了a'-a的运算
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 每一个元素做(a'-a)平方的运算
    sqDiffMat = diffMat ** 2
    # 做(a'-a)^2+(b'-b)^2运算
    sqDistances = sqDiffMat.sum(axis=1)
    # 开根号
    distances = sqDistances ** 0.5
    # 对距离结果排序,排序结果为原集合索引,即dataSet的索引,可对应到labels
    sortedDistances = distances.argsort()
    classCount = {}
    for i in range(kk):
        voteIlabel = labels[sortedDistances[i]]
        # 统计每个标签出现的次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 按value值倒序排列
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回出现次数最多的标签
    return sortedClassCount[0][0]


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


_group, _labels = createDataSet()


# 从文件中读取数据并分析
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# 归一化特征值
def autoNorm(dataSet):
    # 获得每一列最小值
    _minVals = dataSet.min(0)
    # 获得每一列最大值
    maxVals = dataSet.max(0)
    _ranges = maxVals - _minVals
    m = dataSet.shape[0]
    # 每一个元素减去最小值
    normDataSet = dataSet - tile(_minVals, (m, 1))
    # 每一个元素除以差值
    normDataSet = normDataSet / tile(_ranges, (m, 1))
    return normDataSet, _ranges, _minVals


# 测试
def test():
    hoRatio = 0.1
    timeMat, typeLabels = file2matrix('knndata.txt')
    normMat, ranges, minVals = autoNorm(timeMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], typeLabels, 10)
        print("type is:", classifierResult)
        if classifierResult != typeLabels[i]:
            errorCount += 1
    print("error rate is:", (errorCount / float(numTestVecs)))


# 分类函数
def classifyPerson():
    resultList = ['喜欢', '没啥感觉', '讨厌']
    sportInWeek = int(input("一周运动几小时?"))
    gameInWeek = int(input("一周玩电脑游戏几小时?"))
    readInWeek = int(input("一周读书几小时?"))
    timeMat, typeLabels = file2matrix('knndata.txt')
    normMat, ranges, minVals = autoNorm(timeMat)
    inArr = array([sportInWeek, gameInWeek, readInWeek])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, typeLabels, 10)
    print(resultList[classifierResult - 1])


classifyPerson()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(normMat[:, 1], normMat[:, 2], 15.0 * array(typeLabel), 15.0 * array(typeLabel))
# plt.show()
