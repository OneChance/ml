# -*- coding:utf-8 -*-
# 决策树Demo

from math import log
import operator
import treePlotter


def createDataSet():
    _dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    _labels = ['no surfacing', 'flippers']
    return _dataSet, _labels


def calcShannonEnt(_dataSet):
    numEntries = len(_dataSet)
    labelCounts = {}
    for featVec in _dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(_dataSet, axis, value):
    retDataSet = []
    for featVec in _dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def chooseBestFeatureToSplit(_dataSet):
    numFeatures = len(_dataSet[0]) - 1
    baseEntropy = calcShannonEnt(_dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in _dataSet]
        # 某一特征可能值的不重复列表
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            # 得到该特征等于某一可能值的子集合
            subDataSet = splitDataSet(_dataSet, i, value)
            prob = len(subDataSet) / float(len(_dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def createTree(_dataSet, _labels):
    classList = [example[-1] for example in _dataSet]
    # 如果集合中所有样本的分类一样,返回这个分类
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果数据集中只剩下一列(分类标签列),那么将没有特征可用于划分,所以返回出现次数最多的那个分类
    if len(_dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选出此轮最好的划分特征
    bestFeat = chooseBestFeatureToSplit(_dataSet)
    bestFeatLabel = _labels[bestFeat]
    _myTree = {bestFeatLabel: {}}
    del (_labels[bestFeat])
    featValues = [example[bestFeat] for example in _dataSet]
    uniqueVals = set(featValues)
    # 该特征有多少种可能的取值,就划分为多少个子树,每个子树的数据集合,即是_dataSet中特征等于value的样本集合
    for value in uniqueVals:
        subLabels = _labels[:]
        _myTree[bestFeatLabel][value] = createTree(splitDataSet(_dataSet, bestFeat, value), subLabels)
    return _myTree


dataSet, labels = createDataSet()
myTree = createTree(dataSet, labels)

treePlotter.createPlot(myTree)
