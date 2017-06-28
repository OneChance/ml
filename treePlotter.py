# -*- coding:utf-8 -*-
# 绘制决策树的工具

import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    # 参数解释：
    # parentPt:线段末端的坐标
    # centerPt:绘制元素的坐标
    # va ha:绘制的文本框垂直方向上对齐方式 center:坐标parentPt是绘制元素的中心点
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def createPlot():
    # 创建一个白色背景的画布
    fig = plt.figure(1, facecolor='white')
    # 清空画布
    fig.clf()
    # 创建一个一行一列的绘图区
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


# 从字典结构中获取树有多少个节点
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


# 获得树深度
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


# 绘制线段中点上的文本
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    # 绘制文字的时候,适当旋转防重叠
    text = createPlot.ax1.text(xMid, yMid, txtString, rotation=35, va="bottom", ha="left")


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]
    # 判断节点的位置,如果该决策树(可能是决策子树)下有N个叶子节点
    # 判断节点坐标:
    #      第一个叶子节点坐标                    最后一个叶子节点与第一个叶子节点的中间位置
    # (plotTree.xOff+1/numLeafs) + (plotTree.xOff+N/numLeafs-(plotTree.xOff+1/numLeafs))/2
    # 化简后可得下方算式
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    # 绘制判断节点的文字
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    # 向下一层,绘制叶子节点
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            # 每次绘制完叶子节点,坐标水平向右移动一个节点宽度,即得到下一个叶子节点绘制位置
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    # 绘制完叶子节点后,退回上一层,这样可以继续绘制其他兄弟节点
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    # 不显示坐标
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    # 把宽度为1的坐标分成等同于叶子节点的份数,一个叶子节点绘制的空间宽度为1/plotTree.totalW
    # 每次绘制节点时,水平位置上会加上这个长度,由于绘制节点的坐标是节点的中心位置,所以需要向负方向偏移
    # 节点宽度一半的长度,-1/ plotTree.totalW/2 即 -0.5 / plotTree.totalW
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
