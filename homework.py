# -*-coding:utf-8-*-
import lightgbm as lgb
from sklearn.cross_validation import KFold
import datetime
import pandas as pd

trainSet = pd.read_csv('data/zhengqi_train.txt', sep='\t')
testSet = pd.read_csv('data/zhengqi_test.txt', sep='\t')
testSet['target'] = 0
print(trainSet.describe())
print(testSet.describe())


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
    testSet[label] = testSet[label] / Folds

    return testSet[label]
print Model(trainSet=trainSet,testSet=testSet)