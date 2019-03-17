# -*-coding:utf-8-*-
import pandas as pd

from xgboost import XGBRegressor

from sklearn.cross_validation import KFold

import numpy as np

from sklearn.linear_model import Ridge, Lasso

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures

# import  numpy as np

# from matplotlib import pyplot as plt

train = pd.read_csv('data/zhengqi_train.txt', sep='\t')

test = pd.read_csv('data/zhengqi_test.txt', sep='\t')

# 简单特征工程

print(train.describe())

train['new1'] = train['V0'] + train['V1']

test['new1'] = test['V0'] + test['V1']

train['new2'] = train['V2'] + train['V3'] + train['V4']

test['new2'] = test['V2'] + test['V3'] + test['V4']

train_x = train.drop(['target'], axis=1)

Y = train['target']

import lightgbm as lgb
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=2)
p = []
test_err = 0
res = np.zeros((test.shape[0], 5))
for k, (train_index, test_index) in enumerate(kf.split(train_x)):
    x, test_x = train_x.loc[train_index], train_x.loc[test_index]
    y, test_y = Y[train_index], Y[test_index]
    lgb_model = lgb.LGBMRegressor(boosting_type='gbdt',
                                  max_depth=-1,
                                  learning_rate=0.01,
                                  n_estimators=5000,
                                  objective='regression',
                                  )

    lgb_model.fit(x, y,
                  eval_set=[(x, y), (test_x, test_y)],
                  eval_names=['Train', 'Test'],
                  early_stopping_rounds=50,
                  eval_metric='MSE',
                  verbose=50,
                  )
    test_err += mean_squared_error(test_y, lgb_model.predict(test_x))
    res[:, k] += lgb_model.predict(test, num_iteration=lgb_model.best_iteration_)
    print lgb_model.predict(test, num_iteration=lgb_model.best_iteration_)
print test_err / 5
print np.mean(res[:,0:],axis=1)
