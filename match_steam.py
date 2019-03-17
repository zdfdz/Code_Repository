# -*-coding:utf-8-*-
import lightgbm as lgb
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

import numpy as np
import pandas as pd

trainSet = pd.read_csv('data\zhengqi_train.txt', sep='\t')
testSet = pd.read_csv('data\zhengqi_test.txt', sep='\t')

# p = trainSet[(trainSet['V8']<-1)&(trainSet['target']>1)].index
# # 打印一下要删除的行，防止误删
# print p
# # 删除行号为P的一行
# trainSet = trainSet.drop(p,axis=0)

p = trainSet[(trainSet['V1']<-1.7)&(trainSet['target']>0)].index
# 打印一下要删除的行，防止误删
print p
# 删除行号为P的一行
trainSet = trainSet.drop(p,axis=0)
plt.scatter(x=trainSet['V1'],y=trainSet['target'])
plt.show()
