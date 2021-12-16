# # item2vec训练

# user_item_rating = pd.read_csv('data/metadata/user_item_rating.csv')
# user_item_rating.info()
#
# # # 平均分
# # user_item_rating["rating"].mean()
#
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import findspark
findspark.init()

# In[]:

data = pd.read_csv('data/metadata/train_triplets.txt',
                   sep='\t', header=None, names=['user', 'song', 'play_count'])
data.info()
# label编码
user_encoder = LabelEncoder()
data['user'] = user_encoder.fit_transform(data['user'].values)

song_encoder = LabelEncoder()
data['song'] = song_encoder.fit_transform(data['song'].values)


data.info()

# 对于之前的歌曲编码，我们给一个字典，对歌曲和编码进行一一映射
song_labels = dict(zip(song_encoder.classes_, range(len(song_encoder.classes_))))

# 对于那些在之前没有出现过的歌曲，我们直接给一个最大的编码
encoder = lambda x: song_labels[x] if x in song_labels.keys() else len(song_labels)

# 对数据进行labelencoder
track_metadata_df['song_id'] = track_metadata_df['song_id'].apply(encoder)




# In[]:

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("PySpark Item2vec") \
    .getOrCreate()
sc = spark.sparkContext

# pySpark读取数据
df_user_song_merge = spark.read.csv('data/metadata/user_item_rating_group.csv', header=True)
track_metadata_tag_merge = pd.read_csv('data/metadata/track_metadata_tag_merge.csv')
song_title_tags = track_metadata_tag_merge[['song_id', 'title', 'tags']]
song_title_tags.info()


from pyspark.sql import functions as F
from pyspark.sql import types as T
df_user_song_merge = df_user_song_merge.withColumn('songs', F.split(df_user_song_merge.item, ' '))

# 实现word2vec训练与转换
from pyspark.ml.feature import Word2Vec
word2vec = Word2Vec(
    vectorSize=5,
    minCount=0,
    inputCol='songs',
    outputCol='song_2vec')
model = word2vec.fit(df_user_song_merge)

# 不计算user embedding,计算item embedding
model.getVectors().show(3, truncate=False)
model.getVectors().select('word', 'vector') \
    .toPandas() \
    .to_csv('data/metadata/item_embedding.csv', index=False)

# 给定电影，算出最相似10个电影
item_embedding = pd.read_csv('data/metadata/item_embedding.csv')
item_embedding.head(3)