#-*-coding:utf-8-*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss

# df = pd.read_csv("./data/HR.csv")
# df = df.dropna(axis=0,how='any')
# # print df
# sf_s = df["satisfaction_level"]
# print sa_s.value_counts().sort_index()
# plt.title("test")
# plt.xlabel("salary")
# plt.ylabel("money")
# plt.axis([0,4,0,8000])
# plt.xticks(np.arange(len(sa_s.value_counts()))+0.5,sa_s.sort_index())
# # plt.xticks([1,2,3],[sa_s.value_counts().sort_index()])
# # plt.xticks(len(sa_s.value_counts()),sa_s.value_counts())
# plt.bar(np.arange(len(sa_s.value_counts()))+0.5,sa_s.value_counts(),width = 0.5)
#
# f = plt.figure()
# f.add_subplot(1,3,1)
from scipy.stats import f_oneway, ttest_ind

"""
用seaborn来绘制图形
"""
# # 设置显示风格
# sns.set_style(style="whitegrid")
# # 设置内容的字体
# sns.set_context(context="poster",font_scale=0.5)
# # 设置颜色
# sns.set_palette(sns.color_palette("RdBu",n_colors=7))
# # 设置每个柱状的颜色
# sns.countplot(x="salary",hue="department",data=df)
# f= plt.figure()
# f.add_subplot(1,3,1)
# sns.distplot(sf_s,bins=10)
#
# f = plt.figure()
# f.add_subplot(1,3,2)
# sns.distplot(df["last_evaluation"],bins=10)
#
# f = plt.figure()
# f.add_subplot(1,3,3)
# sns.distplot(df["average_monthly_hours"],bins=10)
#
# """
# 箱线图，利用上四分位数和下四分位的范围
# """
# sns.boxplot(y=df["last_evaluation"])
# plt.show()
# """
# 折线图一遍表示数据的变化趋势
# 比如通过对比 工作时长和离职率之类的问题
# """
# # sub_df = df.groupby("time_spend_company").mean()
# # sns.pointplot(sub_df.index,sub_df["left"])
# sns.pointplot(x="time_spend_company",y="left",data=df)
# plt.show()
#
# """
# 饼图
# 用于结构分析
# 计数占比，可以加参数normalize=True：
# """
# lbs = df["department"].value_counts().index
# plt.pie(df["department"].value_counts(normalize=True),labels=lbs,autopct='%1.1f%%')
# plt.show()

# import scipy.stats as ss
# norm_dist = ss.norm.rvs(size=10)
# # norm_dist = [502.8, 502.4 ,499 ,500.3, 504.5 ,498.2,505.6]
# print norm_dist
# print ss.normaltest(norm_dist)
# print "--"*20
#
# # 卡方检验
# print ss.chi2_contingency([[15,95],[85,5]])
# print "--"*20
# # 方差检验 均数是否相同 p<0.05 有很大差异
# print f_oneway([49,50,39,40,43],[28,32,30,26,34],[38,40,45,42,48])
# print "--"*20
# # T检验 均值差别的问题
# print ttest_ind([49,50,39,40,43],[49,5,39,4,0])
# print "--"*20
# # 线性回归问题
# x = np.arange(10).astype(np.float).reshape((10,1))
# y=x*3+4+np.random.random((10,1))
# from sklearn.linear_model import LinearRegression
# reg = LinearRegression()
# print y
# res = reg.fit(x,y)
# y_pred = reg.predict(x)
# print(y_pred)

## pca主程序分析
# data_test = np.array([np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1]),
#                       np.array([2.4,0.7,2.9,2.2,3.0,2.7,1.6,1.1,1.6,0.9])]).T
# print data_test
# print "--"*20
# from sklearn.decomposition import PCA
# lower_data = PCA(n_components=2)
# data_test = lower_data.fit(data_test)
# print data_test
# print "--"*20
# print lower_data.explained_variance_ratio_

# 交叉验证
# df = pd.read_csv("./data/HR.csv")
# df = df.dropna(axis=0,how='any')
# dp_indices = df.groupby(by="department").indices
# # print dp_indices
# sales_values = df["left"].iloc[dp_indices["sales"]].values
# technical_values = df["left"].iloc[dp_indices["technical"]].values
# print ss.ttest_ind(sales_values,technical_values)
# dp_keys = dp_indices.keys()
# # 建立一个矩阵
# dp_t_mat = np.zeros([len(dp_keys),len(dp_keys)])
# for i in range(len(dp_keys)):
#     for j in range(len(dp_keys)):
#         p_value = ss.ttest_ind(df["left"].iloc[dp_indices[dp_keys[i]]].values,
#                                df["left"].iloc[dp_indices[dp_keys[j]]].values)[1]
#         if p_value<0.05:
#             dp_t_mat[i][j]= -1
#         else:
#             dp_t_mat[i][j] = p_value
# sns.heatmap(dp_t_mat, xticklabels=dp_keys, yticklabels=dp_keys)
# plt.show()

#
# # 透视表的的形式观察交叉验证
# piv_tb = pd.pivot_table(df,values="left",index =["promotion_last_5years","salary"],columns=["Work_accident"],aggfunc=np.mean )
# print piv_tb
# sns.heatmap(piv_tb,vmin=0,vmax=1,cmap=sns.color_palette("Reds",n_colors=256))
# plt.show()
"""
分组 和 不纯度检验
"""
# sns.set_context(font_scale=1.5)
# df = pd.read_csv("./data/HR.csv")
# sns.barplot(x="salary",y="left",hue="department",data=df)
# plt.show()

# sl_s = df["satisfaction_level"]
# # sns.barplot(range(len(sl_s)),sl_s.sort_values())
# # plt.show()

"""
相关分析
"""
# df = pd.read_csv("./data/HR.csv")
# sns.heatmap(df.corr(),vmin=-1,vmax=1,cmap=sns.color_palette("RdBu",n_colors=128))
# plt.show()

"""
熵的演练
"""
# s1 = pd.Series(["X1","X1","X2","X2","X2","X2"])
# s2 = pd.Series(["Y1","Y1","Y1","Y1","Y1","Y1"])
# # 熵
# def getEntropy(s):
#     # if not isinstance(s,pd.core.series.Series):
#     #     s=pd.Series(s)
#     # 计算分布
#     prt_ary=pd.groupby(s,by =s).count().values/float(len(s))
#     print prt_ary
#     return -(np.log2(prt_ary)*prt_ary).sum()
# print ("Entropy:",getEntropy(s1))
# print ("Entropy:",getEntropy(s2))
# 条件熵
# def getCondiEntropy(s1,s2):
#     d= dict()
#     for i in list(range(len(s1))):
#         # 准备一个结构体，key是s1的值
#         d[s1[i]] = d.get(s1[i],[])+[s2[i]]
#     return sum([getEntropy(d[k])*len(d[k])/float(len(s1)) for k in d])
# print ("conditEntropy",getCondiEntropy(s1.s2))


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, discriminant_analysis, cross_validation

def load_data():
    diabetes = datasets.load_diabetes()
    return cross_validation.train_test_split(diabetes.data, diabetes.target, test_size=0.25, random_state=0)

def test_LinearRegression(*data):
    X_train, X_test, y_train, y_test = data
    #通过sklearn的linear_model创建线性回归对象
    linearRegression = linear_model.LinearRegression()
    #进行训练
    linearRegression.fit(X_train, y_train)
    #通过LinearRegression的coef_属性获得权重向量,intercept_获得b的值
    print("权重向量:%s, b的值为:%.2f" % (linearRegression.coef_, linearRegression.intercept_))
    #计算出损失函数的值
    print("损失函数的值: %.2f" % np.mean((linearRegression.predict(X_test) - y_test) ** 2))
    #计算预测性能得分
    print("预测性能得分: %.2f" % linearRegression.score(X_test, y_test))

if __name__ == '__main__':
    #获得数据集
    X_train, X_test, y_train, y_test = load_data()
    #进行训练并且输出预测结果
    test_LinearRegression(X_train, X_test, y_train, y_test)

