from deepctr.layers.interaction import FM
import pandas as pd
import sqlite3
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from deepctr.feature_column import SparseFeat, get_feature_names



# In[]
ratings = pd.read_csv("../data/metadata/user_item_rating_all_200w.csv")
track_all_200w = pd.read_csv("../data/metadata/track_all_200w.csv")
# track_all_200w.drop_duplicates(subset=['song'],keep='first',inplace=True)

# 筛选评分记录
# 字典user_playcounts记录每个用户的播放总量
# user_playcounts = {}
# for user, group in track_all_200w.groupby('user'):
#     user_playcounts[user] = group['play_count'].sum()
# temp_user = [user for user in user_playcounts.keys() if user_playcounts[user] > 100]
# temp_playcounts = [playcounts for user, playcounts in user_playcounts.items() if playcounts > 100]
# # track_all_200w = track_all_200w[track_all_200w.user.isin(temp_user)]
#
# ratings = ratings[ratings.user.isin(temp_user)]
# print('歌曲播放量大于100的用户数量占总体用户数量的比例为', str(round(len(temp_user)/len(user_playcounts), 4)*100)+'%')
# print('歌曲播放量大于100的用户产生的播放总量占总体播放总量的比例为', str(round(sum(temp_playcounts) / sum(user_playcounts.values())*100, 4))+'%')
# print('歌曲播放量大于100的用户产生的数据占总体数据的比例为', str(round(len(track_all_200w[track_all_200w.user.isin(temp_user)])/len(track_all_200w)*100, 4))+"%")
#
#
#
# # song_playcounts字典，记录每首歌的播放量
# song_playcounts = {}
# for song, group in track_all_200w.groupby('song'):
#     song_playcounts[song] = group['play_count'].sum()
# temp_song = [song for song in song_playcounts.keys() if song_playcounts[song] > 50]
# temp_playcounts = [playcounts for song, playcounts in song_playcounts.items() if playcounts > 50]
# # track_all_200w = track_all_200w[track_all_200w.song.isin(temp_song)]
#
# ratings = ratings[ratings.song.isin(temp_song)]
# print('播放量大于50的歌曲数量占总体歌曲数量的比例为', str(round(len(temp_song)/len(song_playcounts), 4)*100)+'%')
# print('播放量大于50的歌曲产生的播放总量占总体播放总量的比例为', str(round(sum(temp_playcounts) / sum(song_playcounts.values())*100, 4))+'%')
# print('播放量大于50的歌曲产生的数据占总体数据的比例为', str(round(len(track_all_200w[track_all_200w.song.isin(temp_song)])/len(track_all_200w)*100, 4))+"%")
#
# print("ratings筛选后:****************************************")
# track_all_200w.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)

conn = sqlite3.connect('../db/track_metadata.db')
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
cur.fetchall()
#
# # 获得数据的dataframe
track_metadata_df = pd.read_sql(con=conn, sql='select * from songs')
track_metadata_df = track_metadata_df[['track_id','duration']]

songs = pd.merge(track_all_200w,track_metadata_df,how='inner',on="track_id")
songs = songs[['song','artist_hotttnesss','year','duration']]
# 根据songID去重
songs.drop_duplicates(subset=['song'],keep='first',inplace=True)
# 根据year去除为空，去除为0
songs = songs.dropna(subset=['year'])
songs = songs[songs.year != 0]
print('根据songID去重，去0***********************')
songs.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)
# 去重artist_hotttnesss为空，为0
songs = songs.dropna(subset=['artist_hotttnesss'])
songs = songs[songs.artist_hotttnesss != 0]
print('去重artist_hotttnesss为空，为0***********************')
songs.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)

# 去重duration为空，为0
songs = songs.dropna(subset=['duration'])
songs = songs[songs.duration != 0]
print('duration***********************')
songs.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)



# 评分中year为0的歌曲去除
song_with_year = [song for song in songs.song]
ratings = ratings[ratings.song.isin(song_with_year)]
del(song_with_year)

songs.head()

data = pd.merge(ratings,songs,how='inner',on="song")
n_users = data.user.unique().shape[0]
n_songs = data.song.unique().shape[0]

print('Number of users: {}'.format(n_users))
print('Number of songs: {}'.format(n_songs))
print('Sparsity: {:4.3f}%'.format(float(data.shape[0]) / float(n_users*n_songs) * 100))

del(songs)
del(ratings)
del(track_all_200w)
del(track_metadata_df)

sparse_features = ["song", "user",
                   "artist_hotttnesss", "year", 'duration']
target = ['rating']

# 1.对稀疏特征进行标签编码，对密集特征进行简单变换
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
# 2.统计每个稀疏字段的#唯一特征，并记录密集特征字段名称
fixlen_feature_columns = [SparseFeat(feat, data[feat].max() + 1, embedding_dim=4)
                          for feat in sparse_features]
# 输入到FM的特征 -- 记忆能力，例如：历史点击数据，曝光数据
linear_feature_columns = fixlen_feature_columns
# 输入到Deep部分的特征  -- 泛化能力，例如：视频类型，用户年龄等内容特征
dnn_feature_columns = fixlen_feature_columns
# 获取所有列名
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# 3.为模型生成输入数据
train, test = train_test_split(data, test_size=0.2, random_state=2021)
train_model_input = {name: train[name].values for name in feature_names}
test_model_input = {name: test[name].values for name in feature_names}

# 4.定义模型、训练、预测和评估
# model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
model = FM(linear_feature_columns, dnn_feature_columns, task='regression')
model.compile("adam", "mse", metrics=['mse','mae'], )

history = model.fit(train_model_input, train[target].values,
                    batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
pred_ans = model.predict(test_model_input, batch_size=256)
print("test MSE", round(mean_squared_error(
    test[target].values, pred_ans), 6))
print("test MAE", round(mean_absolute_error(
    test[target].values, pred_ans), 6))

print('结束**********************')

