#-*-coding:utf-8-*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
from sklearn.cross_validation import KFold


df_train = pd.read_csv('data/zhengqi_train.txt', sep='\t')
df_test = pd.read_csv('data/zhengqi_test.txt', sep='\t')
df_test['target'] = 0

p = df_train[(df_train['V1'] < -1.7) & (df_train['target'] > 0)].index
# 打印一下要删除的行，防止误删
print p
# 删除行号为P的一行
df_train = df_train.drop(p, axis=0)


df_train['feature1'] = df_train['V0'] + df_train['V1']
df_train['feature2'] = df_train['V2'] + df_train['V3'] + df_train['V4']
# df_train['feature4'] = df_train['V6'] + df_train['V10']

df_test['feature1'] = df_test['V0'] + df_test['V1']
df_test['feature2'] = df_test['V2'] + df_test['V3'] + df_test['V4']


df = pd.concat((df_train,df_test),axis=0)



df_train = df[df['target']!=0]
df_test = df[df['target']==0]
print df_train.shape,df_test.shape

def Model(trainSet, testSet):
    features = [f for f in trainSet.columns if f not in ['target']]
    label = 'target'
    lgb_model = lgb.LGBMRegressor(
        learning_rate=0.01,
        max_depth=-1,
        n_estimators=5000,
        boosting_type='gbdt',
        random_state=2018,
        objective='regression',
    )
    Folds = 5
    kf = KFold(len(trainSet), n_folds=Folds, shuffle=True, random_state=2018)
    for i, (train_index, test_index) in enumerate(kf):
        print('第{}次训练...'.format(i))
        train = trainSet.iloc[train_index]
        test = trainSet.iloc[test_index]

        lgb_model.fit(
            X=train[features], y=train[label],
            eval_set=[(train[features], train[label]), (test[features], test[label])],
            eval_names=['Train', 'Test'],
            early_stopping_rounds=50,
            eval_metric='MSE',
            verbose=50,
        )
        testSet[label] += lgb_model.predict(testSet[features], num_iteration=lgb_model.best_iteration_)

    # testSet[label] = testSet[label] / Folds

    return testSet[label]/5


sub = Model(trainSet=df_train, testSet=df_test)
sub.to_csv(r'E:\12-17-1.txt', index=False)
print('baseline....')
