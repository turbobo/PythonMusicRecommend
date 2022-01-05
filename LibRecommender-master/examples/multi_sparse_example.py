import time
import numpy as np
import pandas as pd
from libreco.data import split_by_ratio_chrono, DatasetFeat
from libreco.algorithms import DeepFM

# remove unnecessary tensorflow logging
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_WARNINGS"] = "FALSE"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


if __name__ == "__main__":
    data = pd.read_csv("sample_data/sample_movielens_merged.csv",
                       sep=",", header=0)
    train_data, eval_data = split_by_ratio_chrono(data, test_size=0.2)

    # specify complete columns information
    sparse_col = ["sex", "occupation"]
    multi_sparse_col = [["genre1", "genre2", "genre3"]]  # should be list of list
    dense_col = ["age"]
    user_col = ["sex", "age", "occupation"]
    item_col = ["genre1", "genre2", "genre3"]

    train_data, data_info = DatasetFeat.build_trainset(
        train_data=train_data,
        user_col=user_col,
        item_col=item_col,
        sparse_col=sparse_col,
        dense_col=dense_col,
        multi_sparse_col=multi_sparse_col,
        pad_val=["missing"]  # specify padding value
    )
    eval_data = DatasetFeat.build_testset(eval_data)
    print(data_info)
    # do negative sampling, assume the data only contains positive feedback
    train_data.build_negative_samples(data_info, item_gen_mode="random",
                                      num_neg=1, seed=2020)
    eval_data.build_negative_samples(data_info, item_gen_mode="random",
                                     num_neg=1, seed=2222)

    deepfm = DeepFM("ranking", data_info, embed_size=16, n_epochs=2,
                    lr=1e-4, lr_decay=False, reg=None, batch_size=2048,
                    num_neg=1, use_bn=False, dropout_rate=None,
                    hidden_units="128,64,32", tf_sess_config=None,
                    multi_sparse_combiner="sqrtn")  # specify multi_sparse combiner

    deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
               metrics=["loss", "balanced_accuracy", "roc_auc", "pr_auc",
                        "precision", "recall", "map", "ndcg"])

    print("prediction: ", deepfm.predict(user=1, item=2333))
    print("recommendation: ", deepfm.recommend_user(user=1, n_rec=7))
