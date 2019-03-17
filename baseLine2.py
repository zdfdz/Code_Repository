# -*-coding:utf-8-*-
# 数据读取
import pandas as pd
import warnings
import numpy as np

warnings.filterwarnings('ignore')  # 忽略警告（看到一堆警告比较恶心）
train = pd.read_csv('data/zhengqi_train.txt', sep='\t')
test = pd.read_csv('data/zhengqi_test.txt', sep='\t')

p = train[(train['V1'] < -1.7) & (train['target'] > 0)].index
# 打印一下要删除的行，防止误删
print p
# # 删除行号为P的一行
train = train.drop(p, axis=0)

train_x = train.drop(['target'], axis=1)

# train_x['new1'] = train_x['V0'] + train_x['V1']
#
# test['new1'] = test['V0'] + test['V1']

train_x['new2'] = train_x['V2'] + train_x['V3'] + train_x['V4']

test['new2'] = test['V2'] + test['V3'] + test['V4']

all_data = pd.concat([train_x, test])

# 数据观察（可视化）
import seaborn
import matplotlib.pyplot as plt

#
# for col in all_data.columns:
#     seaborn.distplot(train[col])
#     seaborn.distplot(test[col])
#     plt.show()
# 有上面的数据分布来看，特征
# 'V5', 'V17', 'V28', 'V22', 'V11', 'V9'
# 训练集数据与测试集数据分布不一致，会导致模型泛化能力差，采用删除此类特征方法。
all_data.drop(['V5', 'V17', 'V28', 'V22', 'V11', 'V9'], axis=1, inplace=True)

# 数据标准化
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
data_minmax = pd.DataFrame(min_max_scaler.fit_transform(all_data), columns=all_data.columns)

import math

data_minmax['V0'] = data_minmax['V0'].apply(lambda x: math.exp(x))
data_minmax['V1'] = data_minmax['V1'].apply(lambda x: math.exp(x))
data_minmax['V6'] = data_minmax['V6'].apply(lambda x: math.exp(x))
data_minmax['V30'] = np.log1p(data_minmax['V30'])
# train['exp'] = train['target'].apply(lambda x:math.pow(1.5,x)+10)
data_minmax['new1'] = data_minmax['V0'] + data_minmax['V1']

# 针对特征['V0','V1','V6','V30']做数据变换，使得数据符合正态分布
X_scaled = pd.DataFrame(preprocessing.scale(data_minmax), columns=data_minmax.columns)
train_x = X_scaled.ix[0:len(train) - 1]
test = X_scaled.ix[len(train):]

Y = train['target']

# 特征选择
# 通过方差阈值来筛选特征，采用threshold=0.85，剔除掉方差较小，即变化较小的特征删除，因为预测意义小；
#
# 大多数数据已经被标准化到【0，1】之间，通过分析，方差的值域控制为 0.85*（1-0.85）之间有利于特征选择，太大容易删除过多的特征，太小容易保留无效的特征，对预测造成干扰。
# 特征选择
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# 方差
threshold = 0.85
vt = VarianceThreshold().fit(train_x)
# Find feature names
feat_var_threshold = train_x.columns[vt.variances_ > threshold * (1 - threshold)]
train_x = train_x[feat_var_threshold]
test = test[feat_var_threshold]

# 单变量
X_scored = SelectKBest(score_func=f_regression, k='all').fit(train_x, Y)
feature_scoring = pd.DataFrame({
    'feature': train_x.columns,
    'score': X_scored.scores_
})
head_feature_num = 18
feat_scored_headnum = feature_scoring.sort_values('score', ascending=False).head(head_feature_num)['feature']
train_x_head = train_x[train_x.columns[train_x.columns.isin(feat_scored_headnum)]]
X_scaled = pd.DataFrame(preprocessing.scale(train_x), columns=train_x.columns)

# 模型尝试
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb

n_folds = 10


def rmsle_cv(model, train_x_head=train_x_head):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_x_head)
    rmse = -cross_val_score(model, train_x_head, Y, scoring="neg_mean_squared_error", cv=kf)
    return (rmse)


svr = make_pipeline(SVR(kernel='linear'))

line = make_pipeline(LinearRegression())
lasso = make_pipeline(Lasso(alpha=0.0005, random_state=1))
ENet = make_pipeline(ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR1 = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
KRR2 = KernelRidge(alpha=1.5, kernel='linear', degree=2, coef0=2.5)
lgb_model = lgb.LGBMRegressor(
    learning_rate=0.01,
    max_depth=-1,
    n_estimators=5000,
    boosting_type='gbdt',
    random_state=2018,
    objective='regression',
)

rf = RandomForestRegressor(n_estimators=50, max_depth=25, min_samples_split=20,
                           min_samples_leaf=10, max_features='sqrt', oob_score=True, random_state=10)
# KRR3 = KernelRidge(alpha=0.6, kernel='rbf', degree=2, coef0=2.5)
# =============================================================================
# GBoost = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.02,
#                                    max_depth=5, max_features=7,
#                                    min_samples_leaf=15, min_samples_split=10,
#                                    loss='huber', random_state =5)
# =============================================================================

model_xgb = xgb.XGBRegressor(booster='gbtree', colsample_bytree=0.8, gamma=0.1,
                             learning_rate=0.02, max_depth=5,
                             n_estimators=500, min_child_weight=0.8,
                             reg_alpha=0, reg_lambda=1,
                             subsample=0.8, silent=1,
                             random_state=42, nthread=2)

# parameters = {
#             'n_estimators':[300,600,900,1500,2500],
#             #'boosting':'dart',
#             'max_bin':[55,75,95],
#             'num_iterations':[50,100,250,400],
#              # 'max_features':[7,9,11,13],
#               'min_samples_leaf': [15, 25, 35, 45],
#               'learning_rate': [0.01, 0.03, 0.05, 0.1],
#               'num_leaves':[15,31,63],
#
#               'lambda_l2':[0,1]}  # 定义要优化的参数信息
# clf = GridSearchCV( model_lgb, parameters, n_jobs=3,scoring = 'neg_mean_squared_error' )
# clf.fit(train_x,Y)

# print('best n_estimators:', clf.best_params_)
# print('best cv score:', clf.score_)
score = rmsle_cv(lgb_model)
print("\nlgb_model 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(rf)
print("\nrf 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(svr)
print("\nSVR 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
svr.fit(train_x_head, Y)
score = rmsle_cv(line)
print("\nLine 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
line.fit(train_x_head, Y)
score = rmsle_cv(lasso)
print("\nLasso 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
lasso.fit(train_x_head, Y)
score = rmsle_cv(ENet)
print("ElasticNet 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(KRR2)
print("Kernel Ridge2 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
KRR2.fit(train_x_head, Y)
# score = rmsle_cv(KRR3)
# print("Kernel Ridge3 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# =============================================================================
head_feature_num = 18
feat_scored_headnum = feature_scoring.sort_values('score', ascending=False).head(head_feature_num)['feature']
train_x_head2 = train_x[train_x.columns[train_x.columns.isin(feat_scored_headnum)]]
X_scaled = pd.DataFrame(preprocessing.scale(train_x), columns=train_x.columns)
score = rmsle_cv(KRR1, train_x_head2)
print("Kernel Ridge1 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# score = rmsle_cv(GBoost)
# print("Gradient Boosting 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# =============================================================================
head_feature_num = 22
feat_scored_headnum = feature_scoring.sort_values('score', ascending=False).head(head_feature_num)['feature']
train_x_head3 = train_x[train_x.columns[train_x.columns.isin(feat_scored_headnum)]]
X_scaled = pd.DataFrame(preprocessing.scale(train_x), columns=train_x.columns)
score = rmsle_cv(model_xgb, train_x_head3)
print("Xgboost 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
model_xgb.fit(train_x_head, Y)


# =============================================================================
# score = rmsle_cv(model_lgb)
# print("LGBM 得分: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
# =============================================================================
# 简单模型融合
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # 遍历所有模型，你和数据
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        for model in self.models_:
            model.fit(X, y)

        return self

    # 预估，并对预估结果值做average
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        # return 0.85*predictions[:,0]+0.15*predictions[:,1]
        # return 0.7*predictions[:,0]+0.15*predictions[:,1]+0.15*predictions[:,2]
        return np.mean(predictions, axis=1)
    # averaged_models = AveragingModels(models = (lasso,KRR))


# # 对基模型集成后的得分: 0.1161(0.0423)
# averaged_models = AveragingModels(models=(svr,KRR2,model_xgb,lgb_model))
# #对基模型集成后的得分: 0.1141 (0.0417)
averaged_models = AveragingModels(models=(KRR1, KRR2, model_xgb, lasso, lgb_model))
score = rmsle_cv(averaged_models)
print(" 对基模型集成后的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

am = averaged_models.fit(train_x, Y)
pre = am.predict(test)

pred_df = pd.DataFrame(pre)
pred_df = pred_df.astype('float')
pred_df.to_csv(r'E:\new12-10.txt', index=False)
