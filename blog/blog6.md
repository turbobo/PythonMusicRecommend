**生成Embedding的几种方法**

https://github.com/peiss/ant-learn-recsys/blob/master/15.%20%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%BD%93%E5%89%8D%E6%9C%80%E6%B5%81%E8%A1%8C%E7%9A%84Embedding%E7%AE%97%E6%B3%95.ipynb

import findspark
findspark.init("D:\software\spark-2.4.4-bin-hadoop2.7")
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext('local')
spark = SparkSession(sc)
1. 内容向量word2vec
from pyspark.ml.feature import Word2Vec

# Input data: Each row is a bag of words from a sentence or document.
documentDF = spark.createDataFrame([
    ("Hi I heard about Spark".split(" "), ),
    ("I wish Java could use case classes".split(" "), ),
    ("Logistic regression models are neat".split(" "), )
], ["text"])

# Learn a mapping from words to Vectors.
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="result")
model = word2Vec.fit(documentDF)

result = model.transform(documentDF)
for row in result.collect():
    text, vector = row
    print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))
Text: [Hi, I, heard, about, Spark] => 
Vector: [-0.09118835926055908,0.024794609472155574,-0.0023326151072978973]

Text: [I, wish, Java, could, use, case, classes] => 
Vector: [-0.004019131617886679,0.0024854108674584752,0.003071522439963051]

Text: [Logistic, regression, models, are, neat] => 
Vector: [0.05315629169344902,0.02378393579274416,-0.059764140844345094]

把（文档ID，用户词语列表），变成（用户ID，播放电影ID列表），输入到word2vec，就能得到每个电影的Embedding向量

2. 使用Spark ALS的矩阵分解的行为Embedding
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

lines = spark.read.text("D:/workbench/ant-learn-recsys/datas/als/sample_movielens_ratings.txt").rdd
parts = lines.map(lambda row: row.value.split("::"))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                     rating=float(p[2]), timestamp=int(p[3])))
ratings = spark.createDataFrame(ratingsRDD)
(training, test) = ratings.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(maxIter=5, 
          regParam=0.01, 
          userCol="userId", 
          itemCol="movieId", 
          ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(10)
# Generate top 10 user recommendations for each movie
movieRecs = model.recommendForAllItems(10)

# Generate top 10 movie recommendations for a specified set of users
users = ratings.select(als.getUserCol()).distinct().limit(3)
userSubsetRecs = model.recommendForUserSubset(users, 10)
# Generate top 10 user recommendations for a specified set of movies
movies = ratings.select(als.getItemCol()).distinct().limit(3)
movieSubSetRecs = model.recommendForItemSubset(movies, 10)
Root-mean-square error = 1.7636180291295112
training.show(10)
+-------+------+----------+------+
|movieId|rating| timestamp|userId|
+-------+------+----------+------+
|      0|   1.0|1424380312|     3|
|      0|   1.0|1424380312|     5|
|      0|   1.0|1424380312|     6|
|      0|   1.0|1424380312|     8|
|      0|   1.0|1424380312|    11|
|      0|   1.0|1424380312|    13|
|      0|   1.0|1424380312|    15|
|      0|   1.0|1424380312|    19|
|      0|   1.0|1424380312|    20|
|      0|   1.0|1424380312|    21|
+-------+------+----------+------+
only showing top 10 rows

model.itemFactors.show(10, truncate=False)
+---+------------------------------------------------------------------------------------------------------------------------+
|id |features                                                                                                                |
+---+------------------------------------------------------------------------------------------------------------------------+
|0  |[1.2830094, -0.5151567, 0.18020844, -0.71568125, -2.535846, 0.52966636, -0.9128374, -1.1174475, 0.19547723, 0.698292]   |
|10 |[1.0180752, -2.7507377, 1.0930991, -1.134008, -0.5343898, -1.9026369, -1.1077534, -0.8857196, -0.39179775, -0.01721368] |
|20 |[1.4512534, -3.517298, 0.7868661, -1.4977869, -0.24221556, -2.037162, 1.0100238, -1.0118681, -0.3201244, -0.18585977]   |
|30 |[1.4444151, -3.6120615, 1.6103011, 0.17859526, 0.15473363, -0.7841998, 3.4736896, -0.54864204, -0.19071166, 1.2209741]  |
|40 |[1.8021005, -2.4846869, -1.0012007, -0.11796358, -3.8910062, 0.6172575, 0.46259242, 0.20520537, -0.75374764, 0.98922247]|
|50 |[0.85364246, -3.47646, 0.36618742, -1.4283884, -1.0077556, -1.321288, -1.2243408, 0.40804875, 0.07967562, 0.14777525]   |
|60 |[1.1889794, -2.6609316, 1.0438567, -0.45259616, -0.9141676, 0.13169621, -0.44755557, 1.5527409, -1.880157, -0.53658813] |
|70 |[1.7922753, -2.0595644, 0.58770806, -1.4066004, -2.7759798, -1.5907837, 1.471062, -0.028371431, -2.0037463, 0.84014094] |
|80 |[-0.40876412, -1.2445711, 2.2364929, -0.8541729, -2.781932, -0.1874267, 0.9496937, 0.4868828, 1.2105056, -0.40545303]   |
|90 |[2.0153913, -3.951402, 1.3197321, -0.32983437, 1.661446, -2.995254, 1.9340911, -0.087758064, 1.1283636, -0.11234664]    |
+---+------------------------------------------------------------------------------------------------------------------------+
only showing top 10 rows

3. DNN中的Embedding
import tensorflow as tf
import numpy as np
model = tf.keras.Sequential()

`# 注意，这一层是Embedding层
vocab_size:字典大小
embedding_dim:本层的输出大小，也就是生成的embedding的维数
input_length:输入数据的维数，因为输入数据会做padding处理，所以一般是定义的max_length
keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length)`

model.add(tf.keras.layers.Embedding(1000, 64, input_length=10))
model.compile('rmsprop', 'mse')

input_array = np.random.randint(1000, size=(32, 10))
output_array = model.predict(input_array)
print(output_array.shape)
(32, 10, 64)
# 训练完之后，embedding的layer的weights就是embedding向量
model.layers[0].get_weights()
[array([[ 0.00781213,  0.03940525, -0.00024771, ...,  0.03611508,
          0.02551547, -0.03192703],
        [-0.03161997, -0.02198304,  0.03973298, ..., -0.02881846,
         -0.03093203, -0.01212269],
        [ 0.00935531, -0.01970615,  0.03177864, ...,  0.04194124,
         -0.02666444,  0.02423222],
        ...,
        [-0.04647785,  0.01175867,  0.02346585, ..., -0.00246744,
         -0.01744302, -0.00606211],
        [-0.01508133, -0.00510512, -0.02035259, ...,  0.04146155,
         -0.00624609,  0.03074067],
        [ 0.02103826, -0.01366248,  0.01829243, ..., -0.03217832,
          0.02095122, -0.03056069]], dtype=float32)]
 