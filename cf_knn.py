import sqlite3

import matplotlib.pyplot as plt
import numpy as np
# 第三方库
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from surprise import KNNBasic
from surprise import Reader, Dataset, accuracy
from surprise import SVD
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate

# In[]

KNNdata = Dataset.load_builtin('ml-100k')
algo = KNNBasic()
cross_validate(algo, KNNdata, measures = ['MAE','MSE'], cv = 3, verbose = True)


# In[]

user_item_rating = pd.read_csv('data/metadata/user_item_rating_2.csv')

# In[]
# itemCF


# 阅读器
reader = Reader(line_format='user item rating', sep=',')
# 载入数据
raw_data = Dataset.load_from_df(user_item_rating, reader=reader)


algo = KNNBasic()
cross_validate(algo, raw_data, measures = ['MAE','MSE'], cv = 3, verbose = True)


# In[]

# 分割数据集
# kf = StratifiedKFold(n_splits=5)
# 构建模型
kf = KFold(n_splits=5)
# user_based=False 表示以item为基准计算相似度
knn_itemcf = KNNBasic(k=40, sim_options={'user_based': False})
# 训练数据集，并返回rmse误差  --- 10折交叉验证,以避免过拟合和欠拟合
temp_rmse = 0
print("k=40 5折交叉验证 itemCF的平均准确率开始计算：")
for trainset, testset in kf.split(raw_data):
    knn_itemcf.fit(trainset)
    predictions = knn_itemcf.test(testset)
    temp_rmse = temp_rmse + accuracy.rmse(predictions, verbose=True)
# # k=1现在的平均准确率rmse：0.3520
# # k=40现在的平均准确率rmse：0.2761
print("k=40 itemCF的平均准确率rmse：%.6f" % (temp_rmse / 10))


# In[]
# userCF

# 阅读器
reader = Reader(line_format='user song rating', sep=',')
# 载入数据
raw_data = Dataset.load_from_df(user_item_rating, reader=reader)
# 分割数据集
kf = KFold(n_splits=5)
# 构建模型  # user_based=True 表示以user为基准计算相似度
knn_usercf = KNNBasic(k=40, sim_options={'user_based': True})
# 训练数据集，并返回rmse误差
temp_rmse2 = 0
print("k=40 5折交叉验证 userCF的平均准确率开始计算：")
for trainset, testset in kf.split(raw_data):
    knn_usercf.fit(trainset)
    predictions = knn_usercf.test(testset)
    temp_rmse2 = temp_rmse2 + accuracy.rmse(predictions, verbose=True)
print("k=40 userCF的平均准确率rmse：%.6f" % (temp_rmse2 / 10))