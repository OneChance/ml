# -*- coding:utf-8 -*-
# 朴素贝叶斯Demo
from numpy import *
import feedparser


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
            # else:
    # print "this word:%s is not in my vocabulary!" % word
    return returnVec


# 词袋模型,累计单词出现次数
def bagOfWord2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
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
    # vec2Classify是经过setOfWords2Vec处理过后的向量,出现单词的位置上为1(如果是词袋模型,位置上是单词出现次数),
    # 与p1Vec相乘,为1的位置作计算:1*p(wN),即得到该单词出现的概率,因为p1Vec中的概率都做了对数计算,
    # 所以此时的加法运算相当于原先的乘法运算,此处实际是计算:p(w|c1)p(c1),
    # 其中由于假设每个单词具有独立特征,p(w|c1) = p(w0|c1)p(w1|c1)...p(wN|c1)
    # 此处对于公式中的分母p(w)忽略计算,因为只要比较分子大小即可
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


# 文本解析器
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


# _postingList, _classVec = loadDataSet()
# _vocabList = createVocabList(_postingList)

# trainMat = []
# for postinDoc in _postingList:
#    trainMat.append(setOfWords2Vec(_vocabList, postinDoc))

# 0V, p1V, pAb = trainNB0(trainMat, _classVec)

# testEntry = ['love', 'my', 'dalmation']
# testEntry = ['stupid', 'garbage']
# thisDoc = array(setOfWords2Vec(_vocabList, testEntry))

# print testEntry, 'classified as ', classifyNB(thisDoc, p0V, p1V, pAb)

# 查找所有文档中出现的高频词
def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    # 移除高频词,因为这些词多为冗余和结构辅助性词语,也可参考停用词表
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = range(2 * minLen)
    testSet = []
    # 随机抽取20个样本作为测试样本
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWord2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        # 书上此处使用的是词袋模型,不明白为什么,感觉在分类的时候只需要用0的位置剔除不需要计算的概率数据,
        # 而使用1来保留需要计算的概率.
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print 'error rate is :', float(errorCount) / len(testSet)
    return vocabList, p0V, p1V


def getTopWords(ny, sf):
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "--------------------------SF-----------------------------"
    for i in range(5):
        print sortedSF[i][0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "--------------------------NY-----------------------------"
    for i in range(5):
        print sortedNY[i][0]


_ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
_sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')

getTopWords(_ny, _sf)
