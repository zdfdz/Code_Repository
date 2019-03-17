# -*-coding:utf-8-*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 让pd显示全部行数
pd.set_option('display.width', None)
# 导入数据
trainSet = pd.read_csv('data\zhengqi_train.txt', sep='\t')
testSet = pd.read_csv('data\zhengqi_test.txt', sep='\t')
y_train = trainSet.pop('target')
print trainSet.shape, testSet.shape

# 查看每个特征的偏度和峰度与正态对比
from scipy import stats

from scipy.stats import norm, skew  # for some statistics

# for i in range(0, 38):
#     sns.distplot(trainSet['V' + str(i)], fit=norm)
#     sns.distplot(testSet['V' + str(i)], fit=norm)
#     plt.show()

# 查看偏度
# print trainSet['target'].skew()
# 查看峰度
# print trainSet['target'].kurt()

# 相关性分析
# corrmat = trainSet.corr()
# plt.subplots(figsize=(12,9))
# sns.heatmap(corrmat, vmax=0.9, square=True)
# plt.show()

# 查看重要特征和target的关系

# plt.scatter(x=trainSet['V1'],y=trainSet['target'])
# plt.show()

# 剔除训练集异常值,p为行数据
# p = trainSet[(trainSet['V0']<-1)&(trainSet['target']>0.5)].index
# trainSet = trainSet.drop(p,axis=0)
# print trainSet.shape

# 合并数据集
all_df = pd.concat((trainSet, testSet), axis=0)

# 寻找分类向量
# for i in range(0, 38):
#     print all_df['V' + str(i)].astype(str).value_counts().head(3)
# all_df['V9'] = all_df['V9'].astype(str)
# all_df['V17'] = all_df['V17'].astype(str)
# all_df['V22'] = all_df['V22'].astype(str)
# all_df['V24'] = all_df['V24'].astype(str)
# all_df['V28'] = all_df['V28'].astype(str)
# all_df['V33'] = all_df['V33'].astype(str)
# all_df['V34'] = all_df['V34'].astype(str)
# all_df['V35'] = all_df['V35'].astype(str)
# all_dummy_df = pd.get_dummies(all_df)

# # 将其他特征平滑化
# numeric_cols = all_df.columns[all_df.dtypes != 'object']
# # all_dummy_df['V1'] = (all_dummy_df['V1'] - all_dummy_df['V1'].mean())/all_dummy_df['V1'].std()
# # print all_dummy_df
# # print len(numeric_cols)
# feature = [f for f in numeric_cols]
# # for i ,x in enumerate(feature):
# all_dummy_df_mean = all_dummy_df[feature].mean()
# # print all_dummy_df_mean
# all_dummy_df_std = all_dummy_df[feature].std()
# # print all_dummy_df_std
# all_dummy_df[feature] = (all_dummy_df[feature] - all_dummy_df_mean) / all_dummy_df_std
# # print all_dummy_df
# # print all_dummy_df
# all_df = all_dummy_df
# print all_df


# 分离训练集和测试集
trainSet = all_df[0:2888]
testSet = all_df[2888:]

print trainSet.shape

# 选择模型
from sklearn.model_selection import KFold, cross_val_score, train_test_split




# 交叉验证结果
def rmsle_cv(model):
    rmse = np.sqrt(-cross_val_score(model, trainSet.values, y_train, scoring="neg_mean_squared_error", cv=5))
    return rmse


# 模型调参
# 岭回归
from sklearn.linear_model import Ridge
# logspace(a,b,N)
# 把10的a次方到10的b次方区间分成N份
alphas = np.logspace(-1, 1, 20)
test_scores = []
for alpha in alphas:
    clf = Ridge(alpha)
    test_score = rmsle_cv(clf)
    test_scores.append(np.mean(test_score))
plt.plot(alphas, test_scores)
plt.title("岭回归  CV Error")
plt.show()


# 随机森林
from sklearn.ensemble import RandomForestRegressor
max_features = [.1, .3, .5, .7, .9, .99]
test_scores = []
for max_feat in max_features:
    clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
    test_score = rmsle_cv(clf)
    test_scores.append(np.mean(test_score))
plt.plot(max_features, test_scores)
plt.title("Max Features CV Error")
plt.show()

# XGB
import xgboost as xgb
params = [1,2,3,4,5,6]
test_scores = []
for param in params:
    clf = xgb.XGBRegressor(max_depth=param,learning_rate=0.1)
    test_score = rmsle_cv(clf)
    test_scores.append(np.mean(test_score))
plt.plot(params, test_scores)
plt.title("XGB max_depth vs CV Error")
plt.show()

# LightGBM
import lightgbm as lgb
params =  [3,5,8,10]
test_scores = []
for param in params:
    model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=param,
                              learning_rate=0.05, n_estimators=1000)
    test_score = rmsle_cv(model_lgb)
    test_scores.append(np.mean(test_score))
plt.plot(params, test_scores)
plt.title("LightGBM  CV Error")
plt.show()
#
# # 模型选择
#
# lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.001, random_state=1))
#
# ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.001, l1_ratio=.9, random_state=3))
#
# # ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.01, l1_ratio=.9, random_state=3))
#
#
# KRR = KernelRidge(alpha=0.02, kernel='polynomial', degree=2, coef0=2.5)
#
#
#
#
# GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
#                                        max_depth=6, max_features='sqrt',
#                                        min_samples_leaf=15, min_samples_split=10,
#                                        loss='huber', random_state=5)
#
# # GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
# #                                    max_depth=4, max_features='sqrt',
# #                                    min_samples_leaf=15, min_samples_split=10,
# #                                    loss='huber', random_state =5)
#
#
#
# from sklearn.linear_model import Ridge
#
# ridge = Ridge(10)
#
# # # 随机森林
# from sklearn.ensemble import RandomForestRegressor
#
# rft = RandomForestRegressor(n_estimators=200, max_features=0.3)
#
# # # XGB
# import xgboost as xgb
#
# model_xgb = xgb.XGBRegressor(max_depth=4, learning_rate=0.1)
#
# # # LightGBM
# # import lightgbm as lgb
# model_lgb = lgb.LGBMRegressor(learning_rate=0.01,
#                               max_depth=-1,
#                               n_estimators=5000,
#                               boosting_type='gbdt',
#                               random_state=2018,
#                               objective='regression',
#                               )
#
#
# # 模型融合
# class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
#     def __init__(self, models):
#         self.models = models
#
#     # we define clones of the original models to fit the data in
#     def fit(self, X, y):
#         self.models_ = [clone(x) for x in self.models]
#
#         # Train cloned base models
#         for model in self.models_:
#             model.fit(X, y)
#
#         return self
#
#     # Now we do the predictions for cloned models and average them
#     def predict(self, X):
#         predictions = np.column_stack([
#             model.predict(X) for model in self.models_
#         ])
#         return np.mean(predictions, axis=1)
#
# #
# # am = AveragingModels(models=(ridge, rft, model_xgb, model_lgb))
# # print rmsle_cv(am)
# # am.fit(trainSet, y_train)
# # pre_y = am.predict(testSet)
# # pred_df = pd.DataFrame(pre_y)
# # pred_df = pred_df.astype('float')
# # pred_df.to_csv(r'E:\changshi.txt', index=False)
# class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
#     def __init__(self, base_models, meta_model, n_folds=5):
#         self.base_models = base_models
#         self.meta_model = meta_model
#         self.n_folds = n_folds
#
#     # We again fit the data on clones of the original models
#     def fit(self, X, y):
#         self.base_models_ = [list() for x in self.base_models]
#         self.meta_model_ = clone(self.meta_model)
#         kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
#
#         # Train cloned base models then create out-of-fold predictions
#         # that are needed to train the cloned meta-model
#         out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
#         for i, model in enumerate(self.base_models):
#             for train_index, holdout_index in kfold.split(X, y):
#                 instance = clone(model)
#                 self.base_models_[i].append(instance)
#                 instance.fit(X[train_index], y[train_index])
#                 y_pred = instance.predict(X[holdout_index])
#                 out_of_fold_predictions[holdout_index, i] = y_pred
#
#         # Now train the cloned  meta-model using the out-of-fold predictions as new feature
#         self.meta_model_.fit(out_of_fold_predictions, y)
#         return self
#
#     #Do the predictions of all base models on the test data and use the averaged predictions as
#     #meta-features for the final prediction which is done by the meta-model
#     def predict(self, X):
#         meta_features = np.column_stack([
#             np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
#             for base_models in self.base_models_ ])
#         return self.meta_model_.predict(meta_features)
#
# stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
#                                                  meta_model = lasso)
#
#
# am = stacked_averaged_models.fit(trainSet.values,y_train)
# pre_y = am.predict(testSet.values)
# pred_df = pd.DataFrame(pre_y)
# pred_df = pred_df.astype('float')
# pred_df.to_csv(r'E:\lalal.txt', index=False)
