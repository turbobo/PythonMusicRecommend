**Python使用SparkALS矩阵分解实现电影推荐**


背景知识：

协同过滤：简单来说是利用某兴趣相投、拥有共同经验之群体的喜好来推荐用户感兴趣的信息，即群体的智慧
矩阵分解：将（用户、物品、行为）矩阵分解成（用户、隐向量）和（物品，隐向量）两个子矩阵，通过隐向量实现推荐
ALS：交替最小二乘法，先假设U的初始值U(0)，可以根据U(0)可以计算出V(0)，再根据V(0)计算出U(1)，迭代到收敛
演示目标：

实现矩阵分解，得到user embedding和item embedding
对于目标user，近邻搜索得到推荐的item列表（需要去除已看、需要查询电影名称）
延伸：

user embedding自身的搜索，可以实现兴趣相投的人的推荐
item embedding自身的搜索，可以实现相关推荐
import pandas as pd
import numpy as np
import json

import findspark
findspark.init()

from pyspark.sql import SparkSession
1. Pyspark读取CSV数据
spark = SparkSession \
    .builder \
    .appName("PySpark ALS") \
    .getOrCreate()

sc = spark.sparkContext
from pyspark.sql import functions as F
from pyspark.sql import types as T
# 指定excel的解析字段类型
customSchema = T.StructType([
    T.StructField("userId", T.IntegerType(), True),        
    T.StructField("movieId", T.IntegerType(), True),
    T.StructField("rating", T.FloatType(), True),
    T.StructField("timestamp", T.LongType(), True),
])
df = spark.read.csv(
    "./datas/ml-latest-small/ratings_1m.csv", 
    header=True,
    schema=customSchema
)
df.show(5)
+------+-------+------+---------+
|userId|movieId|rating|timestamp|
+------+-------+------+---------+
|     1|   1193|   5.0|978300760|
|     1|    661|   3.0|978302109|
|     1|    914|   3.0|978301968|
|     1|   3408|   4.0|978300275|
|     1|   2355|   5.0|978824291|
+------+-------+------+---------+
only showing top 5 rows

df.select("userId").distinct().count()
6040
df.select("movieId").distinct().count()
3706
df.printSchema()
root
 |-- userId: integer (nullable = true)
 |-- movieId: integer (nullable = true)
 |-- rating: float (nullable = true)
 |-- timestamp: long (nullable = true)

2. 实现SparkALS的矩阵分解
from pyspark.ml.recommendation import ALS
als = ALS(
    maxIter=5, 
    regParam=0.01, 
    userCol="userId", 
    itemCol="movieId", 
    ratingCol="rating",
    coldStartStrategy="drop")

# 实现训练
model = als.fit(df)
保存user embedding
model.userFactors.show(5)
+---+--------------------+
| id|            features|
+---+--------------------+
| 10|[0.59742886, 0.17...|
| 20|[1.309991, 0.5037...|
| 30|[-1.1886241, -0.1...|
| 40|[1.08093, 1.00480...|
| 50|[0.4238868, 0.529...|
+---+--------------------+
only showing top 5 rows

model.userFactors.count()
6040
model.userFactors.select("id", "features") \
           .toPandas() \
           .to_csv('./datas/movielens_sparkals_user_embedding.csv', index=False)
保存item embedding
model.itemFactors.show(5)
+---+--------------------+
| id|            features|
+---+--------------------+
| 10|[0.2586649, 0.356...|
| 20|[0.124496326, -0....|
| 30|[0.95575553, 0.67...|
| 40|[0.31848797, 0.63...|
| 50|[0.45523128, 0.34...|
+---+--------------------+
only showing top 5 rows

model.itemFactors.count()
3706
model.itemFactors.select("id", "features") \
           .toPandas() \
           .to_csv('./datas/movielens_sparkals_item_embedding.csv', index=False)
4. 对于给定用户算出可能最喜欢的10个电影
思路：

查询目标用户的embedding
计算目标用户embedding跟所有movie embedding的sim value
计算用户看过的集合
第2步骤过滤掉看过的集合，然后挑选出前10个电影
# 目标用户ID
target_user_id = 1
4.1 读取多份数据
df_movie = pd.read_csv("./datas/ml-latest-small/movies.csv")
df_movie_embedding = pd.read_csv("./datas/movielens_sparkals_item_embedding.csv")
df_rating = pd.read_csv("./datas/ml-latest-small/ratings.csv")
df_user_embedding = pd.read_csv("./datas/movielens_sparkals_user_embedding.csv")
# embedding从字符串向量化
df_movie_embedding["features"] = df_movie_embedding["features"].map(lambda x : np.array(json.loads(x)))
df_user_embedding["features"] = df_user_embedding["features"].map(lambda x : np.array(json.loads(x)))
4.2 查询用户的embedding
df_user_embedding.head(3)
id	features
0	10	[0.5974288582801819, 0.17486965656280518, 0.04...
1	20	[1.3099910020828247, 0.5037978291511536, 0.260...
2	30	[-1.1886241436004639, -0.13511677086353302, 0....
user_embedding = df_user_embedding[df_user_embedding["id"] == target_user_id].iloc[0, 1]
user_embedding
array([ 0.20444006,  0.44166785, -0.52081656, -0.90068626, -0.23661137,
       -0.87154317, -1.37156034,  1.58559358, -0.5384056 , -2.19185448])
4.3 计算userembedding和所有itemembedding的相似度
df_movie_embedding.head(3)
id	features
0	10	[0.25866490602493286, 0.3560594320297241, 0.15...
1	20	[0.12449632585048676, -0.29282501339912415, -0...
2	30	[0.9557555317878723, 0.6764761805534363, 0.114...
# 余弦相似度
from scipy.spatial import distance
df_movie_embedding["sim_value"] = (
    df_movie_embedding["features"].map(lambda x : 1 - distance.cosine(user_embedding, x)))
df_movie_embedding.head(3)
id	features	sim_value
0	10	[0.25866490602493286, 0.3560594320297241, 0.15...	0.733454
1	20	[0.12449632585048676, -0.29282501339912415, -0...	0.730650
2	30	[0.9557555317878723, 0.6764761805534363, 0.114...	0.554137
4.4 计算用户看过的movieId集合
df_rating.head(3)
userId	movieId	rating	timestamp
0	1	1	4.0	964982703
1	1	3	4.0	964981247
2	1	6	4.0	964982224
# 筛选、查询单列、去重、变成set
watched_ids = set(df_rating[df_rating["userId"] == target_user_id]["movieId"].unique())
len(watched_ids)
232
4.5 筛选出推荐的10个电影ID
df_movie_embedding.head(3)
id	features	sim_value
0	10	[0.25866490602493286, 0.3560594320297241, 0.15...	0.733454
1	20	[0.12449632585048676, -0.29282501339912415, -0...	0.730650
2	30	[0.9557555317878723, 0.6764761805534363, 0.114...	0.554137
# 筛选ID列表
df_target_movieIds = (
    df_movie_embedding[~df_movie_embedding["id"].isin(watched_ids)]
        .sort_values(by="sim_value", ascending=False)
        .head(10)
        [["id", "sim_value"]]
)
df_target_movieIds
id	sim_value
3594	2779	0.958143
2132	2915	0.945573
1298	2013	0.940742
694	3471	0.940604
394	311	0.938639
138	1470	0.938352
2189	3505	0.934250
3662	3499	0.934232
299	3250	0.931460
875	1442	0.930913
4.6 查询ID的电影名称信息展现给用户
pd.merge(
    left=df_target_movieIds,
    right=df_movie,
    left_on="id",
    right_on="movieId"
)[["movieId", "title", "genres", "sim_value"]]
movieId	title	genres	sim_value
0	2779	Heaven Can Wait (1978)	Comedy	0.958143
1	2915	Risky Business (1983)	Comedy	0.945573
2	2013	Poseidon Adventure, The (1972)	Action|Adventure|Drama	0.940742
3	3471	Close Encounters of the Third Kind (1977)	Adventure|Drama|Sci-Fi	0.940604
4	311	Relative Fear (1994)	Horror|Thriller	0.938639
5	3505	No Way Out (1987)	Drama|Mystery|Thriller	0.934250
6	3499	Misery (1990)	Drama|Horror|Thriller	0.934232
7	3250	Alive (1993)	Drama	0.931460
8	1442	Prefontaine (1997)	Drama	0.930913