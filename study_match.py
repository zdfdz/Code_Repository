# -*-coding:utf-8-*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cross_validation import KFold

df_train = pd.read_csv('data/zhengqi_train.txt', sep='\t')
df_test = pd.read_csv('data/zhengqi_test.txt', sep='\t')
# 查看数据集规模。本题中数据集规模较小,应注意过拟合问题
# (1925, 38)(2888, 39)
print df_test.shape, df_train.shape
#
# 查看缺失值比例
train_data_missing = (df_train.isnull().sum() / len(df_train)) * 100
# 没有缺失值
print(train_data_missing)

"""可视化特征处理"""
color = sns.color_palette()
sns.set_style('dark')  # 设置主题模式
# plt.figure(figsize=(20, 16))  # 指定绘图对象宽度和高度
# 得到每个特征的名称
col = df_train.columns.tolist()
# 求相关系数
# 给图指定宽高
plt.figure(figsize=(20, 16))
# 获取每个特征的名称
colnm = df_train.columns.tolist()
mcorr = df_train[colnm].corr()  # 相关系数矩阵，即给出了任意两个变量之间的相关系数
mask = np.zeros_like(mcorr, dtype=np.bool)  # 构造与mcorr同维数矩阵 为bool型
# 设置显示格式为对角线左
mask[np.triu_indices_from(mask)] = True
# cmap = sns.diverging_palette(220, 10, as_cmap=True)  # 返回matplotlib colormap对象
sns.heatmap(mcorr, mask=mask, square=True, annot=True, fmt='0.2f')  # 热力图（看两两相似度）
plt.show()

# 画正太分布图
from scipy import stats
from scipy.stats import norm

sns.distplot(df_train['target'], fit=norm)
(mu, sigma) = norm.fit(df_train['target'])
print('\n mu = {} and sigma = {}\n'.format(mu, sigma))
plt.show()
# 结果有点负偏,通常是大于0.75需要处理,最好是通过log方式处理使其符合正态分布,因为差的不算很大,本题暂且不做处理

# 了解特征值的分布情况
col = df_train.columns.tolist()  # 列表头
fig = plt.figure(figsize=(10, 10))  # 指定绘图对象宽度和高度
# 这里38的意思是在本题有38个特征
for i in range(38):
    plt.subplot(13, 3, i + 1)
    # 画图
    sns.distplot(df_train[col[i]])
    # 显示特征名称
    plt.ylabel(col[i], fontsize=12)
plt.show()

# 计算每个特征的方差,波动小的删之
print np.var(df_train) < 0.3

# 查看重要特征和target的关系

col = ['V0', 'V1', 'V8', 'V27', 'V31']  # 列表头
fig = plt.figure(figsize=(10, 10))  # 指定绘图对象宽度和高度
# 这里38的意思是在本题有38个特征
for i in range(5):
    plt.subplot(3, 3, i + 1)
    # 画图
    plt.scatter(x=df_train[col[i]], y=df_train['target'])
    # 显示特征名称
    plt.xlabel(col[i], fontsize=12)
plt.show()
# #可根据需求查看大图情况
# plt.scatter(x=df_train['V0'],y=df_train['target'])
# plt.scatter(x=df_train['V1'],y=df_train['target'])
# plt.scatter(x=df_train['V8'],y=df_train['target'])
# plt.scatter(x=df_train['V27'],y=df_train['target'])
# plt.scatter(x=df_train['V31'],y=df_train['target'])
# plt.show()

#
# """特征处理"""
# 删除相关性绝对值小于0.2的特征 V9 V13 V14 V17 V18 V19 V21 V22 V25 V26 V28 V29 V30 V32 V33 V34 V35
df_train = df_train.drop(
    ['V14', 'V21', 'V25', 'V26', 'V32', 'V33', 'V34'], axis=1)
df_test = df_test.drop(
    ['V14', 'V21', 'V25', 'V26', 'V32', 'V33', 'V34'], axis=1)

# V1和V27有两个非常明显的离群点,删之
# p = df_train[(df_train['V1'] < -1.7) & (df_train['target'] > 0)].index
# # 打印一下要删除的行，防止误删
# print p
# # # 删除行号为P的一行
# df_train = df_train.drop(p, axis=0)
# plt.scatter(x=df_train['V1'], y=df_train['target'])
# plt.show()


# 重新画一遍图,观察剔除异常值之后的效果
col = ['V0', 'V1', 'V8', 'V27', 'V31']  # 列表头
fig = plt.figure(figsize=(10, 10))  # 指定绘图对象宽度和高度
# 这里38的意思是在本题有38个特征
for i in range(5):
    plt.subplot(3, 3, i + 1)
    # 画图
    plt.scatter(x=df_train[col[i]], y=df_train['target'])
    # 显示特征名称
    plt.xlabel(col[i], fontsize=12)
plt.show()
# 构造新特征
# 因为题目中给的是脱敏数据,只能大概的去分析特征关系
# V0 V1构造原因,与target的corr都为0.87,且两特征相似度极高
df_train['feature1'] = df_train['V0'] + df_train['V1']
df_train['feature2'] = df_train['V2'] + df_train['V3'] + df_train['V4']
# df_train['feature4'] = df_train['V6'] + df_train['V10']

df_test['feature1'] = df_test['V0'] + df_test['V1']
df_test['feature2'] = df_test['V2'] + df_test['V3'] + df_test['V4']
# df_test['feature4'] = df_test['V6'] + df_test['V10']

"""模型选择"""
# 划分feature 和 label
train_x = df_train.drop(['target'], axis=1)
Y = df_train['target']

# 交叉验证结果
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# 对模型进行5折交叉验证
def rmsle_cv(model):
    rmse = np.sqrt(-cross_val_score(model, df_train.values, df_train['target'], scoring="neg_mean_squared_error", cv=5))
    # 返回交叉验证结果
    return rmse

# lightgbm表现相对较好,所以选择lightgbm
import lightgbm as lgb
# KFold,K折交叉验证
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=2)
# 保存每次计算的误差
test_err = 0
# 保存预测结果
res = np.zeros((df_test.shape[0], 5))

# 分五次划分训练集和测试集
for k, (train_index, test_index) in enumerate(kf.split(train_x)):
    x, test_x = train_x.loc[train_index], train_x.loc[test_index]
    y, test_y = Y[train_index], Y[test_index]
    # 创建模型
    lgb_model = lgb.LGBMRegressor(boosting_type='gbdt',
                                  max_depth=-1,
                                  learning_rate=0.01,
                                  n_estimators=5000,
                                  objective='regression',
                                  )
    # 拟合
    lgb_model.fit(x, y,
                  eval_set=[(x, y), (test_x, test_y)],
                  eval_names=['Train', 'Test'],
                  early_stopping_rounds=50,
                  eval_metric='MSE',
                  verbose=50,
                  )
    # 计算均方误差
    test_err += mean_squared_error(test_y, lgb_model.predict(test_x))
    # 保存预测结果
    res[:, k] += lgb_model.predict(df_test, num_iteration=lgb_model.best_iteration_)
    print lgb_model.predict(df_test, num_iteration=lgb_model.best_iteration_)
# 输出K次均方误差的均值
print test_err / 5
# 输出K次预测结果的均值
print np.mean(res[:, 0:], axis=1)
