# -*- coding: utf-8 -*-
# 导入包
import numpy as np
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import pca
from sklearn.ensemble import GradientBoostingRegressor

color = sns.color_palette()  # sns调色板
sns.set_style('dark')  # 设置主题模式
# 1、数据获取
# 2、数据处理
# 3、模型训练
# 4、结果预测
warnings.filterwarnings('ignore')  # 忽略警告（看到一堆警告比较恶心）

'''数据获取'''
df = pd.read_table("data/zhengqi_train.txt")
df_test = pd.read_table("data/zhengqi_test.txt")

# '''数据处理'''
# # 找出相关程度
# plt.figure(figsize=(20, 16))  # 指定绘图对象宽度和高度
# colnm = df.columns.tolist()[:39]  # 列表头
# mcorr = df[colnm].corr()  # 相关系数矩阵，即给出了任意两个变量之间的相关系数
# mask = np.zeros_like(mcorr, dtype=np.bool)  # 构造与mcorr同维数矩阵 为bool型
# mask[np.triu_indices_from(mask)] = True  # 角分线右侧为True
# cmap = sns.diverging_palette(220, 10, as_cmap=True)  # 返回matplotlib colormap对象
# g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')  # 热力图（看两两相似度）
# plt.show()
#
# # 画箱式图
# colnm = df.columns.tolist()[:39]  # 列表头
# fig = plt.figure(figsize=(10, 10))  # 指定绘图对象宽度和高度
# for i in range(38):
#     plt.subplot(13, 3, i + 1)  # 13行3列子图
#     sns.boxplot(df[colnm[i]], orient="v", width=0.5)  # 箱式图
#     plt.ylabel(colnm[i], fontsize=12)
# plt.show()
#
# # 画正太分布图
# sns.distplot(df['target'], fit=norm)
# (mu, sigma) = norm.fit(df['target'])
# print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
# plt.ylabel('Frequency')
# fig = plt.figure()
# res = stats.probplot(df['target'], plot=plt)
# plt.show()
#
# # 查看是否有缺失值（无缺失值 无需补植）
# train_data_missing = (df.isnull().sum() / len(df)) * 100
# print(train_data_missing)
#
# # 分析特性与目标值的相关性 热力图
# corrmat = df.corr()
# f, ax = plt.subplots(figsize=(20, 9))
# sns.heatmap(corrmat, vmax=.8, annot=True)
# plt.show()

# 特征处理
df['new1'] = df['V0'] + df['V1']
df_test['new1'] = df_test['V0'] + df_test['V1']

df['new2'] = df['V2'] + df['V3']
df_test['new2'] = df_test['V2'] + df_test['V3']

df['new3'] = df['V4'] + df['V8']
df_test['new3'] = df_test['V4'] + df_test['V8']

df['new4'] = df['V15'] + df['V16']
df_test['new4'] = df_test['V15'] + df_test['V16']

# 对训练数据处理，分离出特征和标签
X = df.drop('target', axis=1).values
Y = df['target']
X1_test = df_test.values

'''模型训练'''
# 分离出训练集和测试集，并用梯度提升回归训练
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=40)
# 选择模型

from sklearn.model_selection import KFold, cross_val_score, train_test_split


# 交叉验证结果
def rmsle_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X, Y, scoring="neg_mean_squared_error", cv=5))
    return rmse


# 模型调参

# 岭回归
from sklearn.linear_model import Ridge

# logspace(a,b,N)
# 把10的a次方到10的b次方区间分成N份
alphas = np.logspace(-2, 2, 20)
cv_scores = []
# 对参数进行选取
for alpha in alphas:
    # 模型选取Ridge
    clf = Ridge(alpha=alpha)
    # 接受返回的结果
    test_score = rmsle_cv(clf)
    cv_scores.append(np.mean(test_score))
# 画图当参数为alphas时,均方误差为cv_scores
plt.plot(alphas, cv_scores)
# 设置标题
plt.title("Ridge CV Error")
# 显示图片
plt.show()

# 岭回归
from sklearn.linear_model import Lasso

# logspace(a,b,N)
# 把10的a次方到10的b次方区间分成N份
alphas = np.logspace(-3, 0, 50)
cv_scores = []
for alpha in alphas:
    clf = Lasso(alpha)
    test_score = rmsle_cv(clf)
    cv_scores.append(np.mean(test_score))
plt.plot(alphas, cv_scores)
plt.title("Lasso CV Error")
plt.show()

# 随机森林
from sklearn.ensemble import RandomForestRegressor

max_features = [.1, .3, .5, .7, .9, .99]
cv_scores = []
for max_feat in max_features:
    clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
    test_score = rmsle_cv(clf)
    cv_scores.append(np.mean(test_score))
plt.plot(max_features, cv_scores)
plt.title("Max Features  CV Error")
plt.show()

# XGB
import xgboost as xgb

params = [1, 2, 3, 4, 5, 6]
cv_scores = []
for param in params:
    clf = xgb.XGBRegressor(max_depth=param, learning_rate=0.1)
    test_score = rmsle_cv(clf)
    cv_scores.append(np.mean(test_score))
plt.plot(params, cv_scores)
plt.title("XGB max_depth CV Error")
plt.show()

# LightGBM
import lightgbm as lgb
params = [3, 5, 8, 10]
cv_scores = []
for param in params:
    model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=param,
                                  learning_rate=0.01, n_estimators=1000)
    test_score = rmsle_cv(model_lgb)
    cv_scores.append(np.mean(test_score))
plt.plot(params, cv_scores)
plt.title("LightGBM  CV Error")
plt.show()
