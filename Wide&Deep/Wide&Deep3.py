import pandas as pd
import numpy as np
import torch
import sqlite3
from sklearn.model_selection import train_test_split

from pytorch_widedeep import Trainer
from pytorch_widedeep.preprocessing import WidePreprocessor, TabPreprocessor
from pytorch_widedeep.models import Wide, TabMlp, WideDeep
from pytorch_widedeep.metrics import Accuracy
from sklearn.metrics import mean_squared_error




# In[]
path = '../data/metadata/'
ratings = pd.read_csv('../data/metadata/user_item_rating_all_200w.csv')   #, sep=',', header=None, engine='python')
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
conn = sqlite3.connect('../db/track_metadata.db')
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
cur.fetchall()
#
# # 获得数据的dataframe
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

# In[]
# df = pd.read_csv("data/adult/adult.csv.zip")
df = final_df
# df["income_label"] = (df["income"].apply(lambda x: ">50K" in x)).astype(int)
# df.drop("income", axis=1, inplace=True)
# df_train, df_test = train_test_split(df, test_size=0.2, stratify=df.income_label)
df_train, df_test = train_test_split(df, test_size=0.2)

# 准备 wide, crossed, embedding, continuous columns
# wide_cols = [
#     "education",
#     "relationship",
#     "workclass",
#     "occupation",
#     "native-country",
#     "gender",
# ]
wide_cols = ['song_year','artist_hotttnesss','duration','user','song']
# cross_cols = [("education", "occupation"), ("native-country", "occupation")]
cross_cols = [("song_year", "artist_hotttnesss")]
# embed_cols = [
#     ("education", 16),
#     ("workclass", 16),
#     ("occupation", 16),
#     ("native-country", 32),
# ]
embed_cols = [('user',100), ('song',100)]
# cont_cols = ["age", "hours-per-week"]
cont_cols = ['song_year','artist_hotttnesss','duration']
# target_col = "income_label"
target_col = "rating"

# 目标值
target = df_train[target_col].values

# wide侧
wide_preprocessor = WidePreprocessor(wide_cols=wide_cols, crossed_cols=cross_cols)
X_wide = wide_preprocessor.fit_transform(df_train)
wide = Wide(wide_dim=np.unique(X_wide).shape[0], pred_dim=1)

# deep
tab_preprocessor = TabPreprocessor(embed_cols=embed_cols, continuous_cols=cont_cols)
X_tab = tab_preprocessor.fit_transform(df_train)
deeptabular = TabMlp(
    # mlp_hidden_dims=[64, 32],
    mlp_hidden_dims=[256, 128, 64],
    column_idx=tab_preprocessor.column_idx,
    embed_input=tab_preprocessor.embeddings_input,
    continuous_cols=cont_cols,
)

# wide and deep
model = WideDeep(wide=wide, deeptabular=deeptabular)

# 训练模型
# trainer = Trainer(model, objective="binary", metrics=[Accuracy])
trainer = Trainer(model, objective="regression", metrics=[Accuracy])
trainer.fit(
    X_wide=X_wide,
    X_tab=X_tab,
    target=target,
    n_epochs=10,
    batch_size=256,
    # val_split=0.1,
    val_split=0.2,
)

# 预测
X_wide_te = wide_preprocessor.transform(df_test)
X_tab_te = tab_preprocessor.transform(df_test)
preds = trainer.predict(X_wide=X_wide_te, X_tab=X_tab_te)

print("test MSE", round(mean_squared_error(
    df_test[target_col].values, preds), 8))

# 保存和加载

# Option 1: 使用LRHistory回调
trainer.save(path="model_weights", save_state_dict=True)

# Option 2: 另存为任何其他 torch 模型
torch.save(model.state_dict(), "model_weights/wd_model.pt")

# 在此之前，选项1或2是相同的。假设用户已经准备数据并定义新模型组件：

# 1. 建立模型
model_new = WideDeep(wide=wide, deeptabular=deeptabular)
model_new.load_state_dict(torch.load("model_weights/wd_model.pt"))

# 2. 实例化训练器
trainer_new = Trainer(
    model_new,
    objective="binary",
)

# 3. 要么开始拟合，要么直接预测
preds = trainer_new.predict(X_wide=X_wid, X_tab=X_tab)