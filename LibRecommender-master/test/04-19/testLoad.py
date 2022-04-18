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

# 测试加载模型
if __name__ == "__main__":

    # =========================== load model ==============================
    print("\n", "=" * 50, " after load model ", "=" * 50)
    # important to reset graph if model is loaded in the same shell.
    tf.compat.v1.reset_default_graph()
    # load data_info
    data_info = DataInfo.load("D:\IdeaSpace\PythonMusicRecommend\LibRecommender-master\\test\modelSave\deepfm_model3")
    print(data_info)
    # load model, should specify the model name, e.g., DeepFM
    model = DeepFM.load(path="D:\IdeaSpace\PythonMusicRecommend\LibRecommender-master\\test\modelSave\deepfm_model3", model_name="deepfm_model",
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
    print("recommendation: ", model.recommend_user(user=2211, n_rec=7))

    print("recommendation: ", model.recommend_user(user=1, n_rec=7))
    print("recommendation: ", model.recommend_user(user=2211, n_rec=7))