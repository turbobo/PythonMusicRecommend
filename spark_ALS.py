
# -*- coding:utf-8 -*-

# coding=utf-8

# -*- coding=utf-8 -*-

import findspark
# findspark.init("D:\software\spark-2.4.4-bin-hadoop2.7")
findspark.init()

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

sc = SparkContext('local')
spark = SparkSession(sc)


# In[]:

# 使用Spark ALS的矩阵分解的行为Embedding
import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

# lines = spark.read.csv("data/metadata/user_item_rating.csv").rdd
# parts = lines.map(lambda row: row.value.split("::"))
# ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), songId=int(p[1]),
#                                      rating=float(p[2])))
# ratings = spark.createDataFrame(ratingsRDD)
# (training, test) = ratings.randomSplit([0.8, 0.2])

path = './data/metadata/'

# rawUserData = sc.textFile('data/metadata/user_item_rating.csv')
rawUserData = sc.textFile(path+'user_item_rating_all_200w.csv')
rawUserData.count()
rawUserData.first()
rawRatings = rawUserData.map(lambda line:line.split(","))
# rawRatings.take(5)
# ratingsRDD = rawRatings.map(lambda x:(x[0],x[1],x[2]))
ratings = spark.createDataFrame(rawRatings)
(training, test) = ratings.randomSplit([0.8, 0.2])

#在训练数据上使用ALS建立推荐模型
#注意：我们将冷启动策略设置为“下降”，以确保我们不会获得NaN评估指标
# Rank: 对应ALS模型中的因子个数，即矩阵分解出的两个矩阵的新的行/列数，
# 即A≈UVT,k<<m,nA≈UVT,k<<m,n A \approx UV^T , k << m,n中的k。
# MaxIter: 对应运行时的最大迭代次数
# RegParam: 控制模型的正则化过程，从而控制模型的过拟合情况。
als = ALS(maxIter=5,
          regParam=0.01,
          userCol="user",
          itemCol="song",
          ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(training)

#通过计算试验数据的RMSE来评估模型
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# 为每一个用户推荐10个电影
userRecs = model.recommendForAllUsers(10)
# 为每个电影推荐10个用户
movieRecs = model.recommendForAllItems(10)

# 为指定的一组用户生成top10个电影推荐
users = ratings.select(als.getUserCol()).distinct().limit(3)
userSubsetRecs = model.recommendForUserSubset(users, 10)
# 为指定的一组电影生成top10个用户推荐
movies = ratings.select(als.getItemCol()).distinct().limit(3)
movieSubSetRecs = model.recommendForItemSubset(movies, 10)

training.show(10)

model.itemFactors.show(10, truncate=False)
