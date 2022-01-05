import numpy as np
import pandas as pd
from libreco.data import DatasetFeat, DataInfo
from libreco.data import split_by_ratio_chrono
from libreco.algorithms import DeepFM
from libreco.evaluation import evaluate

import os
import tensorflow.compat.v1 as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_WARNINGS"] = "FALSE"
tf.logging.set_verbosity(tf.logging.ERROR)

if __name__ == "__main__":
    col_names = ["user", "item", "label", "time", "sex",
                 "age", "occupation", "genre1", "genre2", "genre3"]
    all_data = pd.read_csv("sample_data/sample_movielens_merged.csv",
                       sep=",", header=0)

    # use first half data as first training part
    first_half_data = all_data[:(len(all_data) // 2)]
    train, test = split_by_ratio_chrono(first_half_data, test_size=0.2)

    sparse_col = ["sex", "occupation", "genre1", "genre2", "genre3"]
    dense_col = ["age"]
    user_col = ["sex", "age", "occupation"]
    item_col = ["genre1", "genre2", "genre3"]
    train_data, data_info = DatasetFeat.build_trainset(train, user_col, item_col,
                                                       sparse_col, dense_col,
                                                       shuffle=False)
    test_data = DatasetFeat.build_testset(test, shuffle=False)
    print(data_info)
    train_data.build_negative_samples(data_info, num_neg=1,
                                      item_gen_mode="random", seed=2020)
    test_data.build_negative_samples(data_info, num_neg=1,
                                     item_gen_mode="random", seed=2222)

    deepfm = DeepFM("ranking", data_info, embed_size=16, n_epochs=2,
                    lr=1e-4, lr_decay=False, reg=None, batch_size=2048,
                    num_neg=1, use_bn=False, dropout_rate=None,
                    hidden_units="128,64,32", tf_sess_config=None)
    deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=test_data,
               metrics=["loss", "balanced_accuracy", "roc_auc", "pr_auc",
                        "precision", "recall", "map", "ndcg"],
               eval_batch_size=8192, k=10, sample_user_num=2048)

    data_info.save(path="model_path")
    deepfm.save(path="model_path", model_name="deepfm_model", manual=True,
                inference_only=False)

    # ===================== load and build new data ========================
    print("\n", "=" * 50, " load and build new data ", "=" * 50)
    tf.reset_default_graph()
    data_info = DataInfo.load("model_path")

    # use second half data as second training part
    second_half_data = all_data[(len(all_data) // 2):]
    train, test = split_by_ratio_chrono(second_half_data, test_size=0.2)

    train_data, data_info = DatasetFeat.build_trainset(
        train, revolution=True, data_info=data_info, merge_behavior=True
    )
    test_data = DatasetFeat.build_testset(test, revolution=True,
                                             data_info=data_info)
    print("new data_info: ", data_info)
    train_data.build_negative_samples(data_info, item_gen_mode="random",
                                      seed=2020)
    test_data.build_negative_samples(data_info, item_gen_mode="random",
                                     seed=2222)

    # ========================== retrain begin =============================
    model = DeepFM("ranking", data_info, embed_size=16, n_epochs=2,
                   lr=1e-4, lr_decay=False, reg=None, batch_size=2048,
                   num_neg=1, use_bn=False, dropout_rate=None,
                   hidden_units="128,64,32", tf_sess_config=None)

    model.rebuild_graph(path="model_path", model_name="deepfm_model",
                        full_assign=True)

    model.fit(train_data, verbose=2, shuffle=True, eval_data=test_data,
              metrics=["loss", "balanced_accuracy", "roc_auc", "pr_auc",
                       "precision", "recall", "map", "ndcg"])

    print("prediction: ", model.predict(user=2211, item=110,
                                        feats={"sex": "mmm",
                                               "genre1": "crime"}))
    print("recommendation: ", model.recommend_user(
        user=2211, n_rec=7, inner_id=False, cold_start="average",
        user_feats=pd.Series({"sex": "F", "occupation": 2, "age": 23}),
        item_data=all_data.iloc[4:10]))

    eval_result = evaluate(model, test, eval_batch_size=8192, k=10,
                           metrics=["roc_auc", "pr_auc", "precision",
                                    "recall", "map", "ndcg"],
                           sample_user_num=2048, neg_sample=True,
                           update_features=False, seed=2222)
    print(eval_result)
