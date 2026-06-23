# -*- coding: utf-8 -*-

# coding: utf-8

import pandas as pd
from libreco.data import DatasetFeat, DataInfo
from libreco.data import split_by_ratio_chrono
from libreco.algorithms import DeepFM
from libreco.evaluation import evaluate

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_WARNINGS"] = "FALSE"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import requests
# import bs4
import redis
import pickle
import sys



# def main(a):
#     print(url)
#     print(sys.argv)
#     print(len(sys.argv))

# 测试加载模型
if __name__ == "__main__":
    # sys.argv[0]代表python程序名，所以列表从1开始读取参数。
    for i in range(1, len(sys.argv)):
        url = sys.argv[i]
    #     main(url)

    # 获取user  转为int
    userId = int (sys.argv[1])
    print("userId:",userId)
    # userId = 2211

    model_path = "D:\IdeaSpace\PythonMusicRecommend\LibRecommender-master\\test\modelSave\deepfm_model_20"

    # =========================== load model ==============================
    # print("\n", "=" * 50, " after load model ", "=" * 50)
    # important to reset graph if model is loaded in the same shell.
    tf.compat.v1.reset_default_graph()
    # load data_info
    data_info = DataInfo.load(model_path)
    print(data_info)
    # load model, should specify the model name, e.g., DeepFM
    model = DeepFM.load(path=model_path, model_name="deepfm_model",
                        data_info=data_info, manual=True)

    # data = pd.read_csv("sample_data/sample_movielens_merged.csv",
    #                    sep=",", header=0)
    # train, test = split_by_ratio_chrono(data, test_size=0.2)
    # eval_result = evaluate(model=model, data=test, eval_batch_size=8192,
    #                        k=10, metrics=["roc_auc", "precision"],
    #                        sample_user_num=2048, neg_sample=True,
    #                        update_features=False, seed=2222)
    # print("eval result: ", eval_result)
    #
    # print("prediction: ", model.predict(user=2211, item=110))


    print("recommendation:", model.recommend_user(user=userId, n_rec=30))
    # print("recommendation:", model.recommend_user(user=1, n_rec=30))

    # recommend = model.recommend_user(user=userId, n_rec=30)
    # recommend_songId = []
    # for i in range(0, len(recommend)):
    #     recommend_songId.append(recommend[i][0])
    #
    # songsSave = pd.read_csv('D:/IdeaSpace/PythonMusicRecommend/LibRecommender-master/test/modelSave/deepfm_model_20/songsSave.csv', sep=',')
    #
    # songsSave = songsSave[songsSave.song.isin(recommend_songId)]
    # songsSave = songsSave[["title", "release", "artist_name", "duration"]]
    # songsSave = songsSave.values.tolist()
    # # songsJson = songsSave.to_json
    # # print(songsSave)
    # # for i in range(len(songsSave)):
    # #     row = songsSave.iloc[i].values.tolist()
    # #     print(row)
    # # return songsSave
    #
    #
    # #连接Redis,选择第二个数据库
    # r = redis.Redis(host='127.0.0.1', port=6379, db= 2)
    # #写数据到Redis
    # idkey = 'songsSave' + str(userId)
    # r.set(idkey, songsSave)

    #hash表数据写入命令hmget，可以一次写入多个键值对
    # r.lpush("lpush",songsJson)
    # r.hmget(idkey,songsJson)
    #写入命令hset，一次只能写入一个键值对
    # r.set("set", songsJson)


    # print("recommendation: ", model.recommend_user(user=1, n_rec=7))
    # print("recommendation: ", model.recommend_user(user=1, n_rec=7))
    print("结束: ")