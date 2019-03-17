#-*-coding:utf-8-*-
from random import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# 散点图
x = np.arange(10)
y = np.random.randn(10)
plt.scatter(x, y, color='blue', marker='*')
plt.show()


# 热点图
matrix_data = np.random.rand(10, 10)
sns.heatmap(data=matrix_data)
plt.show()


# 直方图
plt.title('test')
plt.xlabel('xxx')
plt.ylabel('yyy')
plt.xticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19],['hei','ha','biu','bang','xiu','wang'])
x = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
y_bar = [3, 4, 6, 8, 9, 10, 9, 11, 7, 8]
y_line = [2, 3, 5, 7, 8, 9, 8, 10, 6, 7]
sns.set()
plt.bar(x, y_bar)
# 参数x,y,数轴标点符号,把颜色标记在哪个点上
plt.plot(x, y_line, '-o', color='y')
plt.show()

# 二位散点图自动拟合
iris_data = sns.load_dataset('iris')  # 导入 iris 数据集
print iris_data
# x,y都要是索引里的
sns.lmplot(x='sepal_length', y ='petal_length',hue='species', data=iris_data)
plt.show()


