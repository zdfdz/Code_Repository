#-*-coding:utf-8-*-
import lightgbm as lgb
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

import numpy as np
import pandas as pd
import json

# import json
#
# data = []
# with open('data/cu.json') as f:
#     for line in f:
#         data.append(json.loads(line))
# print data[0]['NcuLink']


df = pd.read_csv('data/name.csv')
print df["NcuLink"][0]






