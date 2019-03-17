# -*-coding:utf-8-*-
import pandas as pd
import numpy as np

train = pd.read_csv('data/zhengqi_train.txt', sep='\t')
test = pd.read_csv('data/zhengqi_test.txt', sep='\t')
train_x = train.drop(['target'], axis=1)
all_data = pd.concat([train_x, test], axis=0)  # 为了统一标准化

all_data.drop(['V5', 'V17', 'V28', 'V22', 'V11', 'V9'], axis=1, inplace=True)

# 数据标准化
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
data_minmax = pd.DataFrame(min_max_scaler.fit_transform(all_data), columns=all_data.columns)

# 偏态修正（考虑到正态分布对于线性模型的预测结果更好）
import math

data_minmax['V0'] = data_minmax['V0'].apply(lambda x: math.exp(x))
data_minmax['V1'] = data_minmax['V1'].apply(lambda x: math.exp(x))
data_minmax['V6'] = data_minmax['V6'].apply(lambda x: math.exp(x))
data_minmax['V30'] = np.log1p(data_minmax['V30'])  # train['exp'] = train['target'].apply(lambda x:math.pow(1.5,x)+10)

X_scaled = pd.DataFrame(preprocessing.scale(data_minmax), columns=data_minmax.columns)
train_x = X_scaled.ix[0:len(train) - 1]
test = X_scaled.ix[len(train):]
Y = train['target']

# 特征选择
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# 方差
threshold = 0.75
vt = VarianceThreshold().fit(train_x)

feat_var_threshold = train_x.columns[vt.variances_ > threshold * (1 - threshold)]
train_x = train_x[feat_var_threshold]
test = test[feat_var_threshold]

# 单变量
X_scored = SelectKBest(score_func=f_regression, k='all').fit(train_x, Y)
feature_scoring = pd.DataFrame({'feature': train_x.columns, 'score': X_scored.scores_})
head_feature_num = 18
feat_scored_headnum = feature_scoring.sort_values('score', ascending=False).head(head_feature_num)['feature']
train_x_head = train_x[train_x.columns[train_x.columns.isin(feat_scored_headnum)]]
X_scaled = pd.DataFrame(preprocessing.scale(train_x), columns=train_x.columns)

from sklearn.decomposition import PCA, KernelPCA

components = 8
pca = PCA(n_components=components).fit(train_x)
pca_variance_explained_df = pd.DataFrame({
    "component": np.arange(1, components + 1),
    "variance_explained": pca.explained_variance_ratio_
})
cols = train_x.columns
train_x = pca.transform(train_x)
test = pca.transform(test[cols])

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
KRR2 = KernelRidge(alpha=1.5, kernel='linear', degree=2,
                   coef0=2.5)  # KRR3 = KernelRidge(alpha=0.6, kernel='rbf', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.02,
                                   max_depth=5, max_features=7,
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=5)
model_xgb = xgb.XGBRegressor(booster='gbtree', colsample_bytree=0.8, gamma=0.1, learning_rate=0.02, max_depth=5,
                             n_estimators=500, min_child_weight=0.8, reg_alpha=0, reg_lambda=1, subsample=0.8, silent=1,
                             random_state=42, nthread=2)

score = rmsle_cv(svr)
print("\nSVR 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
svr.fit(train_x_head, Y)
score = rmsle_cv(line)
print("\nLine 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(lasso)
print("\nLasso 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)
print("ElasticNet 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(KRR2)
print("Kernel Ridge2 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
KRR2.fit(train_x_head,
         Y)
# score = rmsle_cv(KRR3) #print("Kernel Ridge3 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# head_feature_num = 18
# feat_scored_headnum = feature_scoring.sort_values('score', ascending=False).head(head_feature_num)['feature']
# train_x_head2 = train_x[train_x.columns[train_x.columns.isin(feat_scored_headnum)]]
# X_scaled = pd.DataFrame(preprocessing.scale(train_x), columns=train_x.columns)
# score = rmsle_cv(KRR1, train_x_head2)
# print("Kernel Ridge1 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#
# score = rmsle_cv(GBoost)
# print("Gradient Boosting 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



# averaged_models = AveragingModels(models=(svr, KRR2, model_xgb))

# score = rmsle_cv(averaged_models)
# print(" 对基模型集成后的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

