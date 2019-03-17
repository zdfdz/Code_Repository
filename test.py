# -*-coding:utf-8-*-

# 最小二乘法（正规方程）
import np as np
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

X = np.array([1, 5, 7, 9, 13, 16])
Y = np.array([37, 66, 71, 79, 85, 99])
# 维度
print X.shape
plt.scatter(x=X, y=Y, marker='o', c='r')

def hypothesis_function(p, x):
    a, b = p
    y = a * x + b
    return y


def cost_function(p, x, y):
    return hypothesis_function(p, x) - y

p = [50, 100]
from scipy.optimize import leastsq
pa = leastsq(cost_function,p,args=(X,Y))
a,b = pa[0]
print a,b

x = np.linspace(0,16,1000)
y = a*x+b
plt.scatter(x,y,color ="red")
plt.show()
