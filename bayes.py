# -*- coding:utf-8 -*-
# 朴素贝叶斯Demo
from numpy import *


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


# 文档中单词的不重复列表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 词集模型,忽略单词出现的次数
# 从单词列表转换为向量 单词在文档(postingList中的一条记录)中出现为1 不出现为0
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "this word:%s is not in my vocabulary!" % word
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords)  # 为了降低计算概率乘积时0的影响,此处初始化为1
    p1Num = ones(numWords)
    # 非侮辱类文档中单词总数
    p0Denom = 2.0  # 为了降低计算概率乘积时0的影响,此处初始化为2
    # 侮辱类文档中单词总数
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    # 侮辱类文档中每个单词出现的概率
    p1Vect = log(p1Num / p1Denom)  # 防止很小的数计算出现问题
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # vec2Classify是经过setOfWords2Vec处理过后的向量,出现单词的位置上为1,
    # 与p1Vec相乘后,为1的位置计算得到该单词出现的概率,因为p1Vec中的概率都做了对数计算,
    # 所以此时的加法运算相当于原先的乘法运算,此处实际是计算:p(w|c1)p(c1),
    # 其中由于假设每个单词具有独立特征,p(w|c1) = p(w0|c1)p(w1|c1)...p(wN|c1)
    # 此处对于公式中的分母p(w)忽略计算,因为只要比较分子大小即可
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


_postingList, _classVec = loadDataSet()
_vocabList = createVocabList(_postingList)

trainMat = []
for postinDoc in _postingList:
    trainMat.append(setOfWords2Vec(_vocabList, postinDoc))

p0V, p1V, pAb = trainNB0(trainMat, _classVec)

# testEntry = ['love', 'my', 'dalmation']
testEntry = ['stupid', 'garbage']
thisDoc = array(setOfWords2Vec(_vocabList, testEntry))

print testEntry, 'classified as ', classifyNB(thisDoc, p0V, p1V, pAb)
