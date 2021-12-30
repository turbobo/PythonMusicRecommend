import pandas as pd
import sqlite3
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, get_feature_names

# 原因是 tf.keras 会引用独立的 keras 包；
# 而与 tensorflow.python.keras 产生冲突。
# 具体而言，keras 的 input 会生成并使用 node 属性；
# 而 tensorflow.python.keras 里的并不需要。
# 通过 debug 可以发现这个问题，正在考虑提 pull request。

# Keras == 2.3.1 和 tensorflow==2.2.0  2.7.0
# 卸载 Keras==2.7.0 和 tensorflow==2.7.0


# 解决一个简单的二元回归任务
# if __name__ == "__main__":

rating = pd.read_csv("../data/metadata/user_item_rating_all_200w.csv")
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
songs = songs[['song','artist_hotttnesss','year','duration']]
songs.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)
# 去重duration为空，为0
songs = songs.dropna(subset=['duration'])
songs = songs[songs.duration != 0]
print('duration***********************')
songs.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)

data = pd.merge(rating,songs,how='inner',on="song")
n_users = data.user.unique().shape[0]
n_songs = data.song.unique().shape[0]

print('Number of users: {}'.format(n_users))
print('Number of songs: {}'.format(n_songs))
print('Sparsity: {:4.3f}%'.format(float(data.shape[0]) / float(n_users*n_songs) * 100))

del(songs)
del(rating)
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

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# 3.为模型生成输入数据
train, test = train_test_split(data, test_size=0.2, random_state=2020)
train_model_input = {name: train[name].values for name in feature_names}
test_model_input = {name: test[name].values for name in feature_names}

# 4.定义模型、训练、预测和评估
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
model.compile("adam", "mse", metrics=['mse','mae'], )

history = model.fit(train_model_input, train[target].values,
                    batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
pred_ans = model.predict(test_model_input, batch_size=256)
print("test MSE", round(mean_squared_error(
    test[target].values, pred_ans), 6))
print("test MAE", round(mean_absolute_error(
    test[target].values, pred_ans), 6))

print('结束**********************')