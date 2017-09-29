# -*- coding:utf-8 -*-
# 决策树Demo

import operator
import pickle
from math import log

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


def createTree(_dataSet, labels, _nodeLabels, _midLabels):
    _labels = labels[:]
    classList = [example[-1] for example in _dataSet]
    # 如果集合中所有样本的分类一样,返回这个分类
    if classList.count(classList[0]) == len(classList):
        return _nodeLabels[int(classList[0]) - 1]
    # 如果数据集中只剩下一列(分类标签列),那么将没有特征可用于划分,所以返回出现次数最多的那个分类
    if len(_dataSet[0]) == 1:
        return _nodeLabels[int(majorityCnt(classList)) - 1]
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
        _myTree[bestFeatLabel][_midLabels[bestFeatLabel][value]] = createTree(splitDataSet(_dataSet, bestFeat, value),
                                                                              subLabels, _nodeLabels,
                                                                              _midLabels)
    return _myTree


# 分类函数
def classify(inputTree, featLables, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLables.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLables, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    _fr = open(filename)
    return pickle.load(_fr)


# dataSet, labels = createDataSet()
# tree = {}

# try:
#    tree = grabTree('treeclassify.txt')
# except Exception:
#    tree = createTree(dataSet, labels)
#    storeTree(tree, 'treeclassify.txt')

# treePlotter.createPlot(myTree)
# print classify(tree, labels, [1, 0])
# print classify(tree, labels, [1, 1])

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
nodeLabels = ['hard', 'soft', 'no lenses']
midLabels = {'tearRate': {'1': 'reduced', '2': 'normal'}, 'astigmatic': {'1': 'no', '2': 'yes'},
             'prescript': {'1': 'myope', '2': 'hypermetrope'}, 'age': {'1': 'young', '2': 'pre', '3': 'presbyopic'}}
lensesTree = createTree(lenses, lensesLabels, nodeLabels, midLabels)

treePlotter.createPlot(lensesTree)
