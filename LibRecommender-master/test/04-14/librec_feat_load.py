import sqlite3
import time

import pandas as pd
from libreco.algorithms import FM, WideDeep, DeepFM
from libreco.data import DatasetFeat

# 加载数据
from libreco.utils import save_info, save_model_tf_serving
import tensorflow as tf


###### pb模式
# 只有sess中有变量的值，所以保存模型的操作只能在sess内
# pb_dir = "./modelSave/model_pb/"
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     graph_def = tf.get_default_graph().as_graph_def()
#     # 这里是指定要冻结并保存到pb模型中的变量
#     var_list = ["input", "label", "beta", "bias", "output"]   # 如果有name_scope，要写全名，如:"name_scope/beta"
#     constant_graph = tf.graph_util.convert_variables_to_constants(sess, graph_def, var_list)
#     with tf.gfile.FastGFile(pb_dir + "test-model.pb", mode='wb') as f:
#         f.write(constant_graph.SerializeToString())

# 直接从pb获取tensor
# pb_dir = "././modelSave/deepfm_model/deepfm/1/"
# with tf.gfile.FastGFile(pb_dir + "saved_model.pb", "rb") as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())    # 从pb文件中导入信息
#     # 从网络中通过tensor的name获取为变量
#     X, pred = tf.import_graph_def(graph_def, return_elements=["input:0", "output:0"])


# 保存模型3
# 虽然saved_model也支持模型加载，并进行迁移学习。
# 可是不得不说saved_model几乎就是为了部署而生的，
# 因为依靠tf.Serving部署模型时要求模型格式必须是saved_model格式。
# 除此以外saved_model还有另外一个优点就是可以跨语言读取，所以本文也介绍一下这种模式的保存于加载。
# 本文样例的保存在参数设置上会考虑到方便部署。保存好的saved_model结构长这个样子：

# 只有sess中有变量的值，所以保存模型的操作只能在sess内
# version = "1/"
# saved_model_dir = "./saved_model/test-model-dir/"
# builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir + version)
#
# # 构建 signature
# signature = tf.saved_model.signature_def_utils.build_signature_def(
#     # 获取输入输出的信息（shape,dtype等），在部署服务后请求带来的数据会喂到inputs中，服务吐的结果会以outputs的形式返回
#     inputs={"input": tf.saved_model.utils.build_tensor_info(X)},          # 获取输入tensor的信息，这个字典可以有多个key-value对
#     outputs={"output": tf.saved_model.utils.build_tensor_info(pred)},     # 获取输出tensor的信息，这个字典可以有多个key-value对
#     method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME    # 就是'tensorflow/serving/predict'
# )
#
# # 保存到 saved_model
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     builder.add_meta_graph_and_variables(sess,
#                                          tags=[tf.saved_model.tag_constants.SERVING],         # 如果用来部署，就这样写。否则可以写其他，如["test-model"]
#                                          signature_def_map={"serving_default": signature},    # 如果用来部署，字典的key必须是"serving_default"。否则可以写其他
#                                          )
#     builder.save()

mymodel = tf.saved_model.load("D:\IdeaSpace\PythonMusicRecommend\LibRecommender-master\\test\modelSave\deepfm_model\deepfm\1")
print("模型保存结束***************")
print("recommendation: ", mymodel.recommend_user(user=1, n_rec=7))

# reset_state("AutoInt")
# autoint = AutoInt("rating", data_info, embed_size=16, n_epochs=2,
#                   att_embed_size=(8, 8, 8), num_heads=4, use_residual=False,
#                   lr=1e-3, lr_decay=False, reg=None, batch_size=2048,
#                   num_neg=1, use_bn=False, dropout_rate=None,
#                   hidden_units="128,64,32", tf_sess_config=None)
# autoint.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
#             metrics=["rmse", "mae", "r2"])
# print("prediction: ", autoint.predict(user=1, item=2333))
# print("recommendation: ", autoint.recommend_user(user=1, n_rec=7))

# reset_state("DIN")
# din = DIN("rating", data_info, embed_size=16, n_epochs=2,
#           recent_num=10, lr=1e-4, lr_decay=False, reg=None,
#           batch_size=2048, num_neg=1, use_bn=False, dropout_rate=None,
#           hidden_units="128,64,32", tf_sess_config=None, use_tf_attention=True)
# din.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
#         metrics=["rmse", "mae", "r2"])
# print("prediction: ", din.predict(user=1, item=2333))
# print("recommendation: ", din.recommend_user(user=1, n_rec=7))

# print(f"total running time: {(time.perf_counter() - start_time):.2f}")

print("*******************************************结束！*******************************************")