# -*-coding:utf-8-*-
import pandas as pd

from xgboost import XGBRegressor

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

import numpy as np

from sklearn.linear_model import Ridge, Lasso

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures

# import  numpy as np

# from matplotlib import pyplot as plt

train = pd.read_csv('data/zhengqi_train.txt', sep='\t')

test = pd.read_csv('data/zhengqi_test.txt', sep='\t')
# train = train.drop(['V5','V27'],axis=1)
# test = test.drop(['V5','V27'],axis=1)
# 简单特征工程

print(train.describe())

train['new1'] = train['V0'] + train['V1']

test['new1'] = test['V0'] + test['V1']

train['new2'] = train['V2'] + train['V3']

test['new2'] = test['V2'] + test['V3']
#
# train = train.drop(['V0', 'V1', 'V2', 'V3'],axis=1)
# test = test.drop(['V0', 'V1', 'V2', 'V3'],axis=1)
test = test.values
train_x = train.drop(['target'], axis=1)
# train_x = train_x.values
Y = train['target']

# 归一化
from sklearn.preprocessing import MinMaxScaler

mc = MinMaxScaler()
train_x = mc.fit_transform(train_x)
test = mc.transform(test)
Y = train['target']
test_err = 0
y_pred = 0
# print train_x.shape, testSet.shape
from sklearn.decomposition import pca
pca = pca.PCA(n_components=0.95)
pca.fit(train_x)
X_pca = pca.transform(train_x)
X1_pca = pca.transform(test)

kf = KFold(n_splits=5, shuffle=True, random_state=33)

for train_index, test_index in kf.split(train_x):
    # print  train_x[train_index], train_x[test_index]
    x, y = train_x[train_index], Y[train_index]
    x_test, y_test = train_x[test_index], Y[test_index]
    # print x.shape,y.shape
    # print x_test.shape, y_test.shape
    model = XGBRegressor(max_depth=5)
    model.fit(x, y)
    test_err += mean_squared_error(y_test, model.predict(x_test))
    y_pred += model.predict(test)
    print(mean_squared_error(y_test, model.predict(x_test)))

print("平均值为%s" % str(test_err / 5))

pred_df = pd.DataFrame(y_pred/5)
pred_df = pred_df.astype('float')
pred_df.to_csv(r'E:\submit4.txt', index=False)
