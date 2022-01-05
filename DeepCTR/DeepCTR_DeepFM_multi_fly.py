import sqlite3
import numpy as np
import pandas as pd
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from deepctr.feature_column import SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr.models import DeepFM

# 原因是 tf.keras 会引用独立的 keras 包；
# 而与 tensorflow.python.keras 产生冲突。
# 具体而言，keras 的 input 会生成并使用 node 属性；
# 而 tensorflow.python.keras 里的并不需要。
# 通过 debug 可以发现这个问题，正在考虑提 pull request。

# Keras == 2.3.1 和 tensorflow==2.2.0  2.7.0
# 卸载 Keras==2.7.0 和 tensorflow==2.7.0


# 多值输入：具有动态特征散列

ratings = pd.read_csv("../data/metadata/user_item_rating_all_200w.csv")
track_all_200w = pd.read_csv("../data/metadata/track_all_200w.csv")
track_all_200w.drop_duplicates(subset=['song'],keep='first',inplace=True)

conn = sqlite3.connect('../db/track_metadata.db')
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
cur.fetchall()
#
# # 获得数据的dataframe
track_metadata_df = pd.read_sql(con=conn, sql='select * from songs')
track_metadata_df = track_metadata_df[['track_id','duration']]

songs = pd.merge(track_all_200w,track_metadata_df,how='inner',on="track_id")

# 连接tags
print('连接tags***********************')
track_tag_merge = pd.read_csv('../data/metadata/track_tag_merge.txt', sep='\t')
songs = pd.merge(songs,track_tag_merge,how='inner',on="track_id")
songs.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)

songs = songs[['song','artist_hotttnesss','year','duration','tags']]
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

'''
songs = pd.merge(track_all_200w,track_metadata_df,how='inner',on="track_id")
songs = songs[['song','artist_hotttnesss','year','duration']]
songs.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)
# 去重duration为空，为0
songs = songs.dropna(subset=['duration'])
songs = songs[songs.duration != 0]
print('duration***********************')
songs.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)
'''

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


# In[]
sparse_features = ["song", "user",
                   "artist_hotttnesss", "year", 'duration']

data[sparse_features] = data[sparse_features].astype(str)
target = ['rating']

# 1.对稀疏特征和处理序列特征使用动态哈希编码
tags_list = list(map(lambda x: x.split(','), data['tags'].values))
tags_length = np.array(list(map(len, tags_list)))
max_len = max(tags_length)

# 注意：padding=`post`
tags_list = pad_sequences(tags_list, maxlen=max_len, padding='post', dtype=object, value=0).astype(str)

# 2.为每个稀疏字段设置哈希空间，并为序列特征生成特征配置
fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique() * 5, embedding_dim=4, use_hash=True, dtype='string')
                          for feat in sparse_features]
varlen_feature_columns = [
    VarLenSparseFeat(SparseFeat('tags', vocabulary_size=100, embedding_dim=4, use_hash=True, dtype="string"),
                     maxlen=max_len, combiner='mean',
                     )]  # 注意：值0表示序列输入功能的填充
linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# 3.为模型生成输入数据
model_input = {name: data[name] for name in feature_names}
model_input['tags'] = tags_list

# 4.定义模型、训练、预测和评估
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')

model.compile("adam", "mse", metrics=['mse'], )
history = model.fit(model_input, data[target].values,
                    batch_size=256, epochs=10, verbose=2, validation_split=0.2, )

print('结束**********************')