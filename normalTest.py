# -*-coding:utf-8-*-
import numpy as np

x = np.mat([[1, 2.1, 5, 1, 45], [1, 1.4, 3, 2, 40], [1, 1.5, 3, 2, 30], [1, 8.5, 2, 1, 36], [1, 7.3, 2, 1, 40]])
print x
y = np.mat([4.6, 2.3, 3.1, 4.7, 1.5])
print y.T
# 求xTx
ts_x = np.dot(x.T, x)
print  ts_x
if np.linalg.det(ts_x) != 0:
    counter_ts_x = np.linalg.inv(ts_x)
else:
    counter_ts_x = np.linalg.pinv(ts_x)
print counter_ts_x
# (xT*x)-1*x
ctx_x = counter_ts_x * x.T
print ctx_x
calculate_y = ctx_x * y.T
print calculate_y
# [[-35.29425287]
#  [  2.48275862]
#  [  5.42873563]
#  [ 10.01954023]
#  [ -0.05517241]]
