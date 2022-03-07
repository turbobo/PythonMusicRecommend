# LibRecommender Serving Guide

## Introduction

本指南主要介绍如何flask在 LibRecommender 中使用为经过训练的模型提供服务。从服务的角度来看，目前 LibRecommender 中共有三种模型：

基于 KNN 的模型
基于向量的模型
基于张量流的模型。
以下是主要的服务工作流程：

将训练好的模型序列化到磁盘。
加载模型并保存到redis。
运行flask服务器。
向服务器发出 http 请求并获得推荐。
在这里，我们选择不将训练好的模型直接保存到 redis，因为：
1）即使您首先将模型保存到 redis，最终还是会保存到磁盘; 
2）我们尽量保持以下要求主libreco模块尽可能少。

所以在服务过程中，首先应该启动redis服务器：

$ redis-server

保存格式
在 LibRecommender 中，主要的数据序列化格式是JSON而不是 pickle，因为 pickle 比较慢，并且在官方的pickle文档中声明：

警告：该pickle模块对错误或恶意构建的数据不安全。永远不要取消从不受信任或未经身份验证的来源收到的数据。

除了 JSON，基于 tensorflow 构建的模型使用自己的tf.saved_modelAPI 保存，基本上会转换为protocol buffer格式。


## KNN-based model

KNN-based model refers to the classic `userCF` and `itemCF` algorithms, which leverages a similarity matrix to find similar users/items to recommend. Due to the large number of users/items, it is often impractical to store the whole similarity matrix, so here we may only save the most similar `K` neighbors for each user/item. 

Below is an example usage which saves 20 neighbors per item using itermCF. One should also specify model-saving `path` : 

```python
>>> from libreco.algorithms import UserCF, ItemCF
>>> from libreco.data import DatasetPure
>>> from libreco.utils import save_knn
>>> from serving.flask import sim2redis, user_consumed2redis

>>> train_data, data_info = DatasetPure.build_trainset(...)
>>> model = ItemCF(...)
>>> model.fit(...)  # train model
>>> path = "knn_model"  # specify model saving directory
>>> save_knn(path, model, train_data, k=20)  # save model
>>> sim2redis(path)	  # save similarity info to redis
>>> user_consumed2redis(path)  # save user_consumed to redis, in order to prevent from recommending items that the user has consumed
```

```bash
$ cd LibRecommender/serving/flask/knn
$ export FLASK_APP=knn_deploy.py
$ export FLASK_ENV=development  # optional debug mode, never use it in production 
$ flask run  # run flask server

# make requests
$ python knn_request.py --user 1 --k_neighbors 10 --n_rec 10  
$ curl -d '{"user": "1", "n_rec": 10, "k_neighbors": 10}' -X POST http://127.0.0.1:5000/item_cf/recommend
# get item id and score: {'recommend list for user (1)': [[3168, 9.421334058046341], [2538, 8.726857960224152], [505, 8.711400210857391], [530, 7.293927997350693], [1339, 7.1917658150196075], [4270, 7.149620413780212], [601, 7.130850255489349], [3808, 6.961166977882385], [2004, 6.635882019996643], [1300, 6.460416287183762]]}
```



## Vector-based model

Vector-based model relies on the dot product of two vectors to make recommendation, so we only need to save a bunch of vectors. This kind of model includes `SVD`, `SVD++`, `ALS`, `BPR` and `YouTubeMatch`.

In practice, to speed up serving, some ANN(Approximate Nearest Neighbors) libraries are often used to find similar vectors. Here in LibRecommender, we use [faiss](<https://github.com/facebookresearch/faiss>) to do such thing.

Below is an example usage which uses `ALS`. One should also specify model-saving `path` : 

```python
>>> from libreco.algorithms import ALS
>>> from libreco.data import DatasetPure
>>> from libreco.utils import save_vector
>>> from serving.flask import vector2redis, user_consumed2redis, save_faiss_index

>>> train_data, data_info = DatasetPure.build_trainset(...)
>>> model = ALS(...)
>>> model.fit(...)  # train model
>>> path = "vector_model"  # specify model saving directory
>>> save_vector(path, model, train_data)  # save model
>>> vector2redis(path)	  # save vector info to redis
>>> user_consumed2redis(path)  # save user_consumed to redis, in order to prevent from recommending items that the user has consumed
>>> save_faiss_index(path)   # save faiss index if you want to use faiss
```

```bash
$ cd LibRecommender/serving/flask/vector
$ export FLASK_APP=vector_deploy.py
$ export FLASK_ENV=development  # optional debug mode, never use it in production 
$ flask run  # run flask server

# make requests
$ python vector_request.py --user 1 --n_rec 10 --use_faiss false
$ curl -d '{"user": "1", "n_rec": 10, "use_faiss": true}' -X POST http://127.0.0.1:5000/vector/recommend
```



## Tensorflow-based model 

As stated above, tf-based model will typically be saved in `protocol buffer` format. These model mainly contains neural networks, including `Wide & Deep`,  `FM`,  `DeepFM`, `YouTubeRanking` , `AutoInt` , `DIN` . 

We use `tensorflow-serving` to serve tf-based models, and typically it is installed through Docker, see [official page](<https://github.com/tensorflow/serving>) for reference. After successfully starting the docker container, we post request  to the serving model inside the flask app and get the recommendation.

Below is an example usage which uses `DIN`. Since `DIN` makes use of user past interacted items, so we also need to save item sequence to redis. One should also specify model-saving `path` : 

```python
>>> from libreco.algorithms import DIN
>>> from libreco.data import DatasetFeat
>>> from libreco.utils import save_info, save_model_tf_serving
>>> from serving.flask import data_info2redis, user_consumed2redis, seq2redis

>>> train_data, data_info = DatasetFeat.build_trainset(...)
>>> model = DIN(...)
>>> model.fit(...)  # train model

>>> path = "tf_model"  # specify model saving directory
>>> save_info(path, model, train_data, data_info)  # save data_info
>>> save_model_tf_serving(path, model, "din")  # save tf model
>>> data_info2redis(path)	  # save feature info to redis
>>> user_consumed2redis(path)  # save user_consumed to redis, in order to prevent from recommending items that the user has consumed
>>> seq2redis(path)   # save item sequence to redis
```

```bash
$ sudo docker run --rm -t -p 8501:8501 --mount type=bind,source=$(pwd)/tf_model/din,target=/models/din -e MODEL_NAME=din tensorflow/serving   # start tensorflow-serving, make sure that model is in "tf_model/din" directory, or you can change to other directory

$ cd LibRecommender/serving/flask/tf
$ export FLASK_APP=tf_deploy.py
$ export FLASK_ENV=development  # optional debug mode, never use it in production 
$ flask run  # run flask server

# make requests
$ python tf_request.py --user 1 --n_rec 10 --algo din
$ curl -d '{"user": "1", "n_rec": 10, "algo": "din"}' -X POST http://127.0.0.1:5000/din/recommend
```




