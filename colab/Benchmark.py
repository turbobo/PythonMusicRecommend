import pandas as pd
from sklearn.model_selection import KFold
from surprise import KNNBasic
from surprise import Reader, Dataset, accuracy
# from surprise import Reader, Dataset, accuracy
from surprise import SVD
# from surprise import Dataset
# from surprise.model_selection import cross_validate


# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# fastfm2 0.5.2 requires numpy<1.20, but you have numpy 1.21.5 which is incompatible.

# itemCF
#Load data
path = '../data/metadata/'

user_item_rating = pd.read_csv(path+'user_item_rating_all_200w.csv')

# # 阅读器
# reader = Reader(line_format='user item rating', sep=',')
# # 载入数据
# raw_data = Dataset.load_from_df(user_item_rating, reader=reader)
# # 分割数据集
# # kf = StratifiedKFold(n_splits=5)
# # 构建模型
# kf = KFold(n_splits=10)
# # user_based=False 表示以item为基准计算相似度
# knn_itemcf = KNNBasic(k=40, sim_options={'user_based': False})
# # 训练数据集，并返回rmse误差  --- 10折交叉验证,以避免过拟合和欠拟合
# temp_rmse = 0
# print("k=40 10折交叉验证 itemCF的平均准确率开始计算：")
# for trainset, testset in kf.split(raw_data):
#     knn_itemcf.fit(trainset)
#     predictions = knn_itemcf.test(testset)
#     temp_rmse = temp_rmse + accuracy.rmse(predictions, verbose=True)
# # # k=1现在的平均准确率rmse：0.3520
# # # k=40现在的平均准确率rmse：0.2761
# print("k=40 itemCF的平均准确率rmse：%.6f" % (temp_rmse / 10))


# reader = Reader(line_format='user item rating', sep=',')
# # 载入数据
# raw_data = Dataset.load_from_df(user_item_rating, reader=reader)
# # 分割数据集
# kf = KFold(n_splits=10)
# # # 构建模型
# algo = SVD(n_factors=40, biased=True)
# # # 训练数据集，并返回rmse误差
# temp_rmse3 = 0
# print("k=40 10折交叉验证 SVD的平均准确率开始计算：")
# for trainset, testset in kf.split(raw_data):
#     algo.fit(trainset)
#     predictions = algo.test(testset)
#     temp_rmse3 = temp_rmse3 + accuracy.rmse(predictions, verbose=True)
# print("k=40 SVD的平均准确率rmse：%.5f" % (temp_rmse3 / 10))


# 阅读器
reader = Reader(line_format='user item rating', sep=',')
# 载入数据
raw_data = Dataset.load_from_df(user_item_rating, reader=reader)
print(raw_data)
# 分割数据集
# kf = StratifiedKFold(n_splits=5)
# 构建模型
kf = KFold(n_splits=5)
# user_based=False 表示以item为基准计算相似度
knn_itemcf = KNNBasic(k=40, sim_options={'user_based': False})
# 训练数据集，并返回rmse误差  --- 10折交叉验证,以避免过拟合和欠拟合
temp_rmse = 0
temp_mae = 0
print("k=40 5折交叉验证 itemCF的平均准确率开始计算：")
for trainset, testset in kf.split(raw_data):
    knn_itemcf.fit(trainset)
    predictions = knn_itemcf.test(testset)
    temp_rmse = temp_rmse + accuracy.rmse(predictions, verbose=True)
    temp_mae = temp_mae + accuracy.mae(predictions, verbose=True)
# # k=1现在的平均准确率rmse：0.3520
# # k=40现在的平均准确率rmse：0.2761
print("k=40 itemCF的平均准确率rmse：%.6f" % (temp_rmse / 5))
print("k=40 itemCF的平均准确率mae：%.6f" % (temp_mae / 5))
