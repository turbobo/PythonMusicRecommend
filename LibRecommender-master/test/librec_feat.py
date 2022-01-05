import time
import pandas as pd
import sqlite3

from libreco.data import DatasetFeat
from libreco.algorithms import FM, WideDeep, DeepFM, AutoInt, DIN
from keras import initializers

 # 加载数据
path = '../../data/metadata/'
ratings = pd.read_csv(path+'user_item_rating_all_200w.csv')   #, sep=',', header=None, engine='python')
ratings.columns = ['user','song','rating']
ratings.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)

# data
# data = pd.read_csv(path+'track_200w.csv')
data = pd.read_csv(path+'track_all_200w.csv')
# data = data[['user','song','play_count','year','tags']]
data.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)
# data.head(10)
data.astype({'user': 'int32', 'song': 'int32', 'play_count': 'int32', 'year': 'int32'})

# # 字典user_playcounts记录每个用户的播放总量
# user_playcounts = {}
# for user, group in data.groupby('user'):
#     user_playcounts[user] = group['play_count'].sum()
# temp_user = [user for user in user_playcounts.keys() if user_playcounts[user] > 100]
# temp_playcounts = [playcounts for user, playcounts in user_playcounts.items() if playcounts > 100]
# # data = data[data.user.isin(temp_user)]
# ratings = ratings[ratings.user.isin(temp_user)]
# print('歌曲播放量大于100的用户数量占总体用户数量的比例为', str(round(len(temp_user)/len(user_playcounts), 4)*100)+'%')
# print('歌曲播放量大于100的用户产生的播放总量占总体播放总量的比例为', str(round(sum(temp_playcounts) / sum(user_playcounts.values())*100, 4))+'%')
# print('歌曲播放量大于100的用户产生的数据占总体数据的比例为', str(round(len(data[data.user.isin(temp_user)])/len(data)*100, 4))+"%")



# song_playcounts字典，记录每首歌的播放量
# song_playcounts = {}
# for song, group in data.groupby('song'):
#     song_playcounts[song] = group['play_count'].sum()
# temp_song = [song for song in song_playcounts.keys() if song_playcounts[song] > 50]
# temp_playcounts = [playcounts for song, playcounts in song_playcounts.items() if playcounts > 50]
# # data = data[data.song.isin(temp_song)]
# ratings = ratings[ratings.song.isin(temp_song)]
# print('播放量大于50的歌曲数量占总体歌曲数量的比例为', str(round(len(temp_song)/len(song_playcounts), 4)*100)+'%')
# print('播放量大于50的歌曲产生的播放总量占总体播放总量的比例为', str(round(sum(temp_playcounts) / sum(song_playcounts.values())*100, 4))+'%')
# print('播放量大于50的歌曲产生的数据占总体数据的比例为', str(round(len(data[data.song.isin(temp_song)])/len(data)*100, 4))+"%")

# data['duration']=data['duration'].apply(lambda x : int(x))
# data.astype({'duration': 'int32'})
print("data:****************************************")
data.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)


#songs
conn = sqlite3.connect('../../db/track_metadata.db')
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
cur.fetchall()
#
# # 获得歌曲数据的详细信息
track_metadata_df = pd.read_sql(con=conn, sql='select * from songs')
track_metadata_df = track_metadata_df[['track_id','duration']]

songs = pd.merge(data,track_metadata_df,how='inner',on="track_id")
songs = songs[['song','artist_hotttnesss','year','duration']]
songs = songs.rename(columns={'year': 'song_year'})
# 根据songID去重
songs.drop_duplicates(subset=['song'],keep='first',inplace=True)
# 根据year去除为空，去除为0
songs = songs.dropna(subset=['song_year'])
# songs = songs[songs.song_year != 0]
print('根据songID去重，去0***********************')
songs.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)
# 去重artist_hotttnesss为空，为0
songs = songs.dropna(subset=['artist_hotttnesss'])
# songs = songs[songs.artist_hotttnesss != 0]
print('去重artist_hotttnesss为空，为0***********************')
songs.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)

# 去重duration为空，为0
songs = songs.dropna(subset=['duration'])
# songs = songs[songs.duration != 0]
print('duration***********************')
songs.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)



# 评分中year为0的歌曲去除
# song_with_year = [song for song in songs.song]
# ratings = ratings[ratings.song.isin(song_with_year)]
# del(song_with_year)

songs.head()

# song_with_year.head(10)
# song_with_year.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)
#Users
# users = pd.read_csv(path+'users.dat', sep='::', header=None, engine='python')
# users.columns = ['userId','Gender','Age','Occupation','Zip-code']
# users = users.drop('Zip-code', axis=1)
# user没有附加信息
# users = data[['user']]

#Data quality
print('Duplicated rows in ratings file: ' + str(ratings.duplicated().sum()))

# n_users = ratings.userId.unique().shape[0]
# n_movies = ratings.movieId.unique().shape[0]
n_users = ratings.user.unique().shape[0]
n_songs = ratings.song.unique().shape[0]

print('Number of users: {}'.format(n_users))
print('Number of songs: {}'.format(n_songs))
print('Sparsity: {:4.3f}%'.format(float(ratings.shape[0]) / float(n_users*n_songs) * 100))

# 评分和歌曲连接
# final_df = ratings.merge(songs, left_on='song', right_on='song', how='inner')
# final_df = pd.merge(ratings,songs,how='left',on="song")
final_df = pd.merge(ratings,songs,how='inner',on="song")
# user、song不按照播放量筛选：1648992条数据
final_df.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)

n_users = final_df.user.unique().shape[0]
n_songs = final_df.song.unique().shape[0]

print('Number of users: {}'.format(n_users))
print('Number of songs: {}'.format(n_songs))
print('Sparsity: {:4.3f}%'.format(float(final_df.shape[0]) / float(n_users*n_songs) * 100))

final_df.head(10)

# 释放内存
del(songs)
del(ratings)
del(track_metadata_df)

"""###筛选"""

# remove unnecessary tensorflow logging
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_WARNINGS"] = "FALSE"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def reset_state(name):
    tf.compat.v1.reset_default_graph()
    print("\n", "=" * 30, name, "=" * 30)

"""### 定义函数"""

import math
import numpy as np
from sklearn.model_selection import train_test_split


def random_split(data, test_size=None, multi_ratios=None, shuffle=True,
                 filter_unknown=True, pad_unknown=False, seed=42):
    ratios, n_splits = _check_and_convert_ratio(test_size, multi_ratios)
    if not isinstance(ratios, list):
        ratios = list(ratios)

    # if we want to split data in multiple folds,
    # then iteratively split data based on modified ratios
    train_data = data.copy()
    split_data_all = []
    for i in range(n_splits - 1):
        size = ratios.pop(-1)
        ratios = [r / math.fsum(ratios) for r in ratios]
        train_data, split_data = train_test_split(train_data,
                                                  test_size=size,
                                                  shuffle=shuffle,
                                                  random_state=seed)
        split_data_all.insert(0, split_data)
    split_data_all.insert(0, train_data)  # insert final fold of data

    if filter_unknown:
        split_data_all = _filter_unknown_user_item(split_data_all)
    elif pad_unknown:
        split_data_all = _pad_unknown_user_item(split_data_all)
    return split_data_all


def _filter_unknown_user_item(data_list):
    train_data = data_list[0]
    unique_values = dict(user=set(train_data.user.tolist()),
                         item=set(train_data.item.tolist()))

    split_data_all = [train_data]
    for i, test_data in enumerate(data_list[1:], start=1):
        # print(f"Non_train_data {i} size before filtering: {len(test_data)}")
        out_of_bounds_row_indices = set()
        for col in ["user", "item"]:
            for j, val in enumerate(test_data[col]):
                if val not in unique_values[col]:
                    out_of_bounds_row_indices.add(j)

        mask = np.arange(len(test_data))
        test_data_clean = test_data[~np.isin(
            mask, list(out_of_bounds_row_indices))]
        split_data_all.append(test_data_clean)
        # print(f"Non_train_data {i} size after filtering: "
        #      f"{len(test_data_clean)}")
    return split_data_all


def _pad_unknown_user_item(data_list):
    train_data, test_data = data_list
    n_users = train_data.user.nunique()
    n_items = train_data.item.nunique()
    unique_users = set(train_data.user.tolist())
    unique_items = set(train_data.item.tolist())

    split_data_all = [train_data]
    for i, test_data in enumerate(data_list[1:], start=1):
        test_data.loc[~test_data.user.isin(unique_users), "user"] = n_users
        test_data.loc[~test_data.item.isin(unique_items), "item"] = n_items
        split_data_all.append(test_data)
    return split_data_all


def split_by_ratio(data, order=True, shuffle=False, test_size=None,
                   multi_ratios=None, filter_unknown=True, pad_unknown=False,
                   seed=42):
    np.random.seed(seed)
    assert ("user" in data.columns), "data must contains user column"
    ratios, n_splits = _check_and_convert_ratio(test_size, multi_ratios)

    n_users = data.user.nunique()
    user_indices = data.user.to_numpy()
    user_split_indices = _groupby_user(user_indices, order)

    cum_ratios = np.cumsum(ratios).tolist()[:-1]
    split_indices_all = [[] for _ in range(n_splits)]
    for u in range(n_users):
        u_data = user_split_indices[u]
        u_data_len = len(u_data)
        if u_data_len <= 3:   # keep items of rare users in trainset
            split_indices_all[0].extend(u_data)
        else:
            u_split_data = np.split(u_data, [
                round(cum * u_data_len) for cum in cum_ratios
            ])
            for i in range(n_splits):
                split_indices_all[i].extend(list(u_split_data[i]))

    if shuffle:
        split_data_all = tuple(
            np.random.permutation(data[idx]) for idx in split_indices_all)
    else:
        split_data_all = list(data.iloc[idx] for idx in split_indices_all)

    if filter_unknown:
        split_data_all = _filter_unknown_user_item(split_data_all)
    elif pad_unknown:
        split_data_all = _pad_unknown_user_item(split_data_all)
    return split_data_all


def split_by_num(data, order=True, shuffle=False, test_size=1,
                 filter_unknown=True, pad_unknown=False, seed=42):
    np.random.seed(seed)
    assert ("user" in data.columns), "data must contains user column"
    assert isinstance(test_size, int), "test_size must be int value"
    assert 0 < test_size < len(data), "test_size must be in (0, len(data))"

    n_users = data.user.nunique()
    user_indices = data.user.to_numpy()
    user_split_indices = _groupby_user(user_indices, order)

    train_indices = []
    test_indices = []
    for u in range(n_users):
        u_data = user_split_indices[u]
        u_data_len = len(u_data)
        if u_data_len <= 3:   # keep items of rare users in trainset
            train_indices.extend(u_data)
        elif u_data_len <= test_size:
            train_indices.extend(u_data[:-1])
            test_indices.extend(u_data[-1:])
        else:
            k = test_size
            train_indices.extend(u_data[:(u_data_len-k)])
            test_indices.extend(u_data[-k:])

    if shuffle:
        train_indices = np.random.permutation(train_indices)
        test_indices = np.random.permutation(test_indices)

    split_data_all = (data.iloc[train_indices], data.iloc[test_indices])
    if filter_unknown:
        split_data_all = _filter_unknown_user_item(split_data_all)
    elif pad_unknown:
        split_data_all = _pad_unknown_user_item(split_data_all)
    return split_data_all

# 修改
def split_by_ratio_chrono(data, order=True, shuffle=False, test_size=None,
                          multi_ratios=None, seed=42):
    assert all([
        "user" in data.columns,
        # "time" in data.columns
    ]), "data must contains user"

    # data.sort_values(by=["time"], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return split_by_ratio(**locals())


def split_by_num_chrono(data, order=True, shuffle=False, test_size=1, seed=42):
    assert all([
        "user" in data.columns,
        # "time" in data.columns
    ]), "data must contains user and time column"

    # data.sort_values(by=["time"], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return split_by_num(**locals())


def _groupby_user(user_indices, order):
    sort_kind = "mergesort" if order else "quicksort"
    users, user_position, user_counts = np.unique(user_indices,
                                                  return_inverse=True,
                                                  return_counts=True)
    user_split_indices = np.split(np.argsort(user_position, kind=sort_kind),
                                  np.cumsum(user_counts)[:-1])
    return user_split_indices


def _check_and_convert_ratio(test_size, multi_ratios):
    if not test_size and not multi_ratios:
        raise ValueError("must provide either 'test_size' or 'multi_ratios'")

    elif test_size is not None:
        assert isinstance(test_size, float), "test_size must be float value"
        assert 0.0 < test_size < 1.0, "test_size must be in (0.0, 1.0)"
        ratios = [1 - test_size, test_size]
        return ratios, 2

    elif isinstance(multi_ratios, (list, tuple)):
        assert len(multi_ratios) > 1, (
            "multi_ratios must at least have two elements")
        assert all([r > 0.0 for r in multi_ratios]), (
            "ratios should be positive values")
        if math.fsum(multi_ratios) != 1.0:
            ratios = [r / math.fsum(multi_ratios) for r in multi_ratios]
        else:
            ratios = multi_ratios
        return ratios, len(ratios)

    else:
        raise ValueError("multi_ratios should be list or tuple")

"""### 连接"""

tart_timestart_time = time.perf_counter()
# data = pd.read_csv("sample_data/sample_movielens_merged.csv",
#                    sep=",", header=0)
# train_data, eval_data = split_by_ratio_chrono(data, test_size=0.2)

n_users = final_df.user.unique().shape[0]
n_songs = final_df.song.unique().shape[0]

print('Number of users: {}'.format(n_users))
print('Number of songs: {}'.format(n_songs))
print('Sparsity: {:4.3f}%'.format(float(final_df.shape[0]) / float(n_users*n_songs) * 100))

final_df = final_df.rename(columns={'song': 'item'})
final_df = final_df.rename(columns={'rating': 'label'})
train_data, eval_data = split_by_ratio_chrono(final_df, test_size=0.2)

# 指定完整的列信息
# sparse_col = ["sex", "occupation", "genre1", "genre2", "genre3"]
# dense_col = ["age"]
# user_col = ["sex", "age", "occupation"]
# item_col = ["genre1", "genre2", "genre3"]

# train_data, data_info = DatasetFeat.build_trainset(
#     train_data, user_col, item_col, sparse_col, dense_col
# )
# eval_data = DatasetFeat.build_testset(eval_data)
# print(data_info)

# specify complete columns information
# "user", "song","artist_hotttnesss", "year", 'duration'

sparse_col = ["song_year"]
dense_col = ["artist_hotttnesss", "duration"]
user_col = ["user"]
item_col = ["item", "artist_hotttnesss", "song_year", "duration"]

train_data, data_info = DatasetFeat.build_trainset(
    train_data, user_col, item_col, sparse_col, dense_col
)
eval_data = DatasetFeat.build_testset(eval_data)
print(data_info)

# hidden_dim隐藏维度基本上是每层中的节点数（例如在多层感知器中）
# embed_size嵌入大小告诉您特征向量的大小（模型使用嵌入的词作为输入）
reset_state("FM")
fm = FM("rating", data_info, embed_size=16, n_epochs=10,
        lr=0.001, lr_decay=False, reg=None, batch_size=256,
        num_neg=1, use_bn=True, dropout_rate=None, tf_sess_config=None)
fm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
        metrics=["rmse", "mae", "r2"])
# print("prediction: ", fm.predict(user=1, item=2333))
# print("recommendation: ", fm.recommend_user(user=1, n_rec=7))

reset_state("Wide_Deep")
wd = WideDeep("rating", data_info, embed_size=16, n_epochs=10,
              lr={"wide": 0.01, "deep": 0.001}, lr_decay=False, reg=None,
              batch_size=256, num_neg=1, use_bn=False, dropout_rate=None,
              hidden_units="256,128,64", tf_sess_config=None)
wd.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
        metrics=["rmse", "mae", "r2"])
# print("prediction: ", wd.predict(user=1, item=2333))
# print("recommendation: ", wd.recommend_user(user=1, n_rec=7))

reset_state("DeepFM")
deepfm = DeepFM("rating", data_info, embed_size=16, n_epochs=10,
                lr=0.001, lr_decay=False, reg=None, batch_size=256,
                num_neg=1, use_bn=False, dropout_rate=None,
                hidden_units="256,128,64", tf_sess_config=None)
deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
            metrics=["rmse", "mae", "r2"])
# print("prediction: ", deepfm.predict(user=1, item=2333))
# print("recommendation: ", deepfm.recommend_user(user=1, n_rec=7))

# reset_state("AutoInt")
# autoint = AutoInt("rating", data_info, embed_size=16, n_epochs=2,
#                   att_embed_size=(8, 8, 8), num_heads=4, use_residual=False,
#                   lr=1e-3, lr_decay=False, reg=None, batch_size=2048,
#                   num_neg=1, use_bn=False, dropout_rate=None,
#                   hidden_units="128,64,32", tf_sess_config=None)
# autoint.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
#             metrics=["rmse", "mae", "r2"])
# print("prediction: ", autoint.predict(user=1, item=2333))
# print("recommendation: ", autoint.recommend_user(user=1, n_rec=7))

# reset_state("DIN")
# din = DIN("rating", data_info, embed_size=16, n_epochs=2,
#           recent_num=10, lr=1e-4, lr_decay=False, reg=None,
#           batch_size=2048, num_neg=1, use_bn=False, dropout_rate=None,
#           hidden_units="128,64,32", tf_sess_config=None, use_tf_attention=True)
# din.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
#         metrics=["rmse", "mae", "r2"])
# print("prediction: ", din.predict(user=1, item=2333))
# print("recommendation: ", din.recommend_user(user=1, n_rec=7))

# print(f"total running time: {(time.perf_counter() - start_time):.2f}")