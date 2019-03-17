#-*-coding:utf-8-*-

import numpy as np
#用自定义函数loadDataSet创建实验文档样本
def loadDataSet():
    """
    创建数据集
    :return: 文档包含单词的列表postingList, 分类标签列表classVec
    """
    #用列表postingList创建文档列表
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], #[0,0,1,1,1......]
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 列表classVec创建标签，1代表侮辱性文字，0代表正常言论
    classVec = [0, 1, 0, 1, 0, 1]
    # 返回文档列表postingList及标签classVec
    return postingList, classVec
listOPosts,listClasses =loadDataSet()
print listOPosts


def createVocabList(dataSet):
    """
    获取所有单词的集合
    :param dataSet: 数据集
    :return: 所有单词的集合(即不含重复元素的单词列表)
    """
    vocabSet =  set()
    for document in dataSet:
        # 操作符 | 用于求两个集合的并集
        vocabSet=set(document)|vocabSet
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    """
    遍历查看该单词是否出现，出现该单词则将该单词置1,否则该单词置0
    :param vocabList: 所有单词集合列表
    :param inputSet: 输入数据集
    :return: 匹配列表[0,1,0,1...]，其中 1与0 表示词汇表中的单词是否出现在输入的数据集中
    """
    # 创建一个和词汇表vocabList等长的向量returnVec,向量中每一元素都为0
    returnVec = [0]*len(vocabList)# [0,0......]
    #用变量word遍历输入文档inputSet中的所有单词
    for word in inputSet:
        # 如果单词在词汇表vocabList中
        if word in vocabList:
            # 则将输出文档向量中的值设为1
            returnVec[vocabList.index(word)]=1
        else:
            # 否则输出“单词不在词汇表中”,%用作格式化字符串
            print("the word:%s is not in my Vocabulary!"% word)
    # 返回文档向量returnVec
    return returnVec

listOPosts,listClasses =loadDataSet()
myVocabList = createVocabList(listOPosts)

print myVocabList
print setOfWords2Vec(myVocabList, listOPosts[0])