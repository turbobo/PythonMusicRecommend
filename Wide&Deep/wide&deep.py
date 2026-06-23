import numpy as np
import tensorflow as tf
import pandas as pd
import random
import math
import re

from sklearn import preprocessing
from os import path, listdir
from sklearn.datasets import load_svmlight_files
from sklearn.model_selection import train_test_split
from sklearn import metrics
# from tensorflow.contrib import layers

from sklearn import metrics

import time
import datetime

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf

# In[]
# 数据准备

# 定义输入样本格式
_CSV_COLUMNS = [
    'rating', 'tags', 'artist_hotttnesss'
    # , 'education', 'education_num',
    # 'marital_status', 'occupation', 'relationship', 'race', 'gender',
    # 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    # 'income_bracket'
]
_CSV_COLUMN_DEFAULTS = [[0], [''], [0]
    # , [''], [0], [''], [''], [''], [''], [''],
    #                     [0], [0], [0], [''], ['']
                        ]
_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}

# 构建一组deep特征列和wide特征列
def build_model_columns():
    # 1. 特征处理，包括：连续特征、离散特征、转换特征、交叉特征等

    # 连续特征 （其中在Wide和Deep组件都会用到）
    rating = tf.feature_column.numeric_column('rating')
    artist_hotttnesss = tf.feature_column.numeric_column('artist_hotttnesss')
    #歌曲年份、

    # capital_gain = tf.feature_column.numeric_column('capital_gain')
    # capital_loss = tf.feature_column.numeric_column('capital_loss')
    # hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    # 离散特征
    # tag = tf.feature_column.categorical_column_with_vocabulary_list(
    #     'education', [
    #         'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
    #         'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
    #         '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])
    #
    # marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
    #     'marital_status', [
    #         'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
    #         'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])
    #
    # relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    #     'relationship', [
    #         'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
    #         'Other-relative'])
    #
    # workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    #     'workclass', [
    #         'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
    #         'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

    # 离散hash bucket特征
    # occupation = tf.feature_column.categorical_column_with_hash_bucket(
    #     'occupation', hash_bucket_size=1000
    # )

    # 有4480个标签，希望分成100个分类
    tags = tf.feature_column.categorical_column_with_hash_bucket(
        'tags', hash_bucket_size=1000
    )

    # 特征Transformations
    # age_buckets = tf.feature_column.bucketized_column(
    #     age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    # )

    # 2. 设定Wide层特征
    """
    # Wide部分使用了规范化后的连续特征、离散特征、交叉特征
    """
    # 基本特征列
    base_columns = [
        # 全是离散特征
        tags
    ]

    # 交叉特征列
    crossed_columns = [
        # 评分和歌曲标签交叉
        tf.feature_column.crossed_column(
            ['rating', 'tags'], hash_bucket_size=1000)
        # ,
        # tf.feature_column.crossed_column(
        #     [age_buckets, 'education', 'occupation'], hash_bucket_size=1000
        # )
    ]

    # wide特征列
    wide_columns = base_columns + crossed_columns

    # 3. 设定Deep层特征
    """
    Deep层主要针对离散特征进行处理，其中处理方式有：
    1. Sparse Features -> Embedding vector -> 串联(连续特征)，其中Embedding Values随机初始化。
    2. 另外一种处理离散特征的方法是：one-hot和multi-hot representation. 此方法适用于低维度特征，其中embedding是通用的做法
    其中：采用embedding_column(embedding)和indicator_column(multi-hot)API
    """
    # deep特征列
    deep_columns = [
        rating,
        artist_hotttnesss,
        # capital_gain,
        # capital_loss,
        # hours_per_week,
        # tf.feature_column.indicator_column(workclass),
        # tf.feature_column.indicator_column(education),
        # tf.feature_column.indicator_column(marital_status),
        # tf.feature_column.indicator_column(relationship),

        # embedding特征
        # tf.feature_column.embedding_column(occupation, dimension=8)
        tf.feature_column.embedding_column(tags, dimension=8)
    ]
    return wide_columns, deep_columns

# In[]
# 定义输入

def input_fn(data_file, num_epochs, shuffle, batch_size):
    """为Estimator创建一个input function"""
    #assert判断，为False时执行后面语句
    assert tf.gfile.Exists(data_file), "{0} not found.".format(data_file)
    def parse_csv(line):
        print("Parsing", data_file)
        # tf.decode_csv会把csv文件转换成Tensor。其中record_defaults用于指明每一列的缺失值用什么填充。
        columns = tf.decode_csv(line, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        #pop函数提取label
        labels = features.pop('income_bracket')
        # tf.equal(x, y) 返回一个bool类型Tensor， 表示x == y, element-wise
        return features, tf.equal(labels, '>50K')
    dataset = tf.data.TextLineDataset(data_file).map(parse_csv, num_parallel_calls=4)
    '''
    使用 tf.data.Dataset.map，我们可以很方便地对数据集中的各个元素进行预处理。
    map接收一个函数，Dataset中的每个元素都会被当作这个函数的输入，并将函数返回值作为新的Dataset
    因为输入元素之间时独立的，所以可以在多个 CPU 核心上并行地进行预处理。
    num_parallel_calls 参数的最优值取决于你的硬件、训练数据的特质（比如：它的 size、shape）、
    map 函数的计算量 和 CPU 上同时进行的其它处理。比较简单的一个设置方法是：将 num_parallel_calls 设置为 CPU 的核心数。
    例如，CPU 有四个核心时，将 num_parallel_calls 设置为 4 将会很高效。
    相反，如果 num_parallel_calls 大于 CPU 的核心数，将导致低效的调度，导致输入管道的性能下降。
    
    也可以设置shuffle
    shuffle的功能为打乱dataset中的元素，它有一个参数buffersize，表示打乱时使用的buffer的大小，建议舍的不要太小，一般是1000
    '''

    dataset = dataset.repeat(num_epochs)
    #repeat的功能就是将整个序列重复多次，主要用来处理机器学习中的epoch，假设原先的数据是一个epoch，使用repeat(2)就可以将之变成2个epoch
    dataset = dataset.batch(batch_size)
    '''
    batch是机器学习中批量梯度下降法(Batch Gradient Descent, BGD)的概念，
    在每次梯度下降的时候取batch-size的数据量做平均来获取梯度下降方向，
    例如我们将batch-size设为2，那么每次iterator都会得到2个数据
    '''
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


# In[]
# 模型准备

# Wide & Deep 模型
def build_estimator(model_dir, model_type):
    """Build an estimator appropriate for the given model type."""
    wide_columns, deep_columns = build_model_columns()
    hidden_units = [100, 50]

    # 创建一个tf.estimator.RunConfig以确保模型在CPU上运行，CPU的训练速度比该模型的GPU快
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))

    if model_type == 'wide':
        return tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns,
            config=run_config)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            config=run_config)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_units,
            config=run_config)


# In[]
# 模型训练

# 模型路径
model_type = 'widedeep'
model_dir = '../data/metadata/'

# Wide & Deep 联合模型
model = build_estimator(model_dir, model_type)


# ## 4）模型训练

# In[11]:

# 训练参数
train_epochs = 10
batch_size = 5000
train_file = '/Users/admin/Desktop/model/推荐算法/widedeep/adult.data'
test_file = '/Users/admin/Desktop/model/推荐算法/widedeep/adult.test'

# 6. 开始训练
for n in range(train_epochs):
    # 模型训练
    model.train(input_fn=lambda: input_fn(train_file, train_epochs, True, batch_size))
    # 模型评估
    results = model.evaluate(input_fn=lambda: input_fn(test_file, 1, False, batch_size))
    # 打印评估结果
    print("Results at epoch {0}".format((n+1) * train_epochs))
    print('-'*30)
    for key in sorted(results):
        print("{0:20}: {1:.4f}".format(key, results[key]))






# 博客链接  https://blog.csdn.net/Andy_shenzl/article/details/105222609