# -*- coding:utf-8 -*-
# k临近算法Demo

from numpy import *
import operator

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
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # 返回出现次数最多的标签
    return sortedClassCount[0][0]


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


_group, _labels = createDataSet()

print classify0([0, 0], _group, _labels, 3)
