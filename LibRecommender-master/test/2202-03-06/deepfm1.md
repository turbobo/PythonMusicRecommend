deepfm = DeepFM("rating", data_info, embed_size=16, n_epochs=80,
lr=0.001, lr_decay=False, reg=None, batch_size=256,
num_neg=1, use_bn=False, dropout_rate=0.5,
hidden_units="256,256,256", tf_sess_config=None)



D:\Develop\Python37\python.exe C:\Users\73556\AppData\Roaming\JetBrains\IntelliJIdea2021.2\plugins\python\helpers\pydev\pydevconsole.py --mode=client --port=59270
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\IdeaSpace\\PythonMusicRecommend', 'D:/IdeaSpace/PythonMusicRecommend'])
PyDev console: starting.
Python 3.7.2 (tags/v3.7.2:9a3ffc0492, Dec 23 2018, 23:09:28) [MSC v.1916 64 bit (AMD64)] on win32
runfile('D:/IdeaSpace/PythonMusicRecommend/LibRecommender-master/test/librec_feat.py', wdir='D:/IdeaSpace/PythonMusicRecommend/LibRecommender-master/test')
2022-03-06 12:00:22.333328: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-03-06 12:00:22.333443: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
WARNING:tensorflow:From D:\Develop\Python37\lib\site-packages\tensorflow\python\compat\v2_compat.py:107: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2052374 entries, 0 to 2052373
Data columns (total 3 columns):
#   Column  Non-Null Count    Dtype
---  ------  --------------    -----  
0   user    2052374 non-null  int64  
1   song    2052374 non-null  int64  
2   rating  2052374 non-null  float64
dtypes: float64(1), int64(2)
memory usage: 47.0 MB
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2052374 entries, 0 to 2052373
Data columns (total 10 columns):
#   Column              Non-Null Count    Dtype
---  ------              --------------    -----  
0   user                2052374 non-null  int64  
1   song                2052374 non-null  int64  
2   play_count          2052374 non-null  int64  
3   track_id            2052374 non-null  object
4   title               2052369 non-null  object
5   release             2052374 non-null  object
6   artist_name         2052374 non-null  object
7   artist_familiarity  2052374 non-null  float64
8   artist_hotttnesss   2052374 non-null  float64
9   year                2052374 non-null  int64  
dtypes: float64(2), int64(4), object(4)
memory usage: 156.6+ MB
data:****************************************
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2052374 entries, 0 to 2052373
Data columns (total 10 columns):
#   Column              Non-Null Count    Dtype
---  ------              --------------    -----  
0   user                2052374 non-null  int64  
1   song                2052374 non-null  int64  
2   play_count          2052374 non-null  int64  
3   track_id            2052374 non-null  object
4   title               2052369 non-null  object
5   release             2052374 non-null  object
6   artist_name         2052374 non-null  object
7   artist_familiarity  2052374 non-null  float64
8   artist_hotttnesss   2052374 non-null  float64
9   year                2052374 non-null  int64  
dtypes: float64(2), int64(4), object(4)
memory usage: 156.6+ MB
根据songID去重，去0***********************
<class 'pandas.core.frame.DataFrame'>
Int64Index: 193683 entries, 0 to 2052373
Data columns (total 4 columns):
#   Column             Non-Null Count   Dtype
---  ------             --------------   -----  
0   song               193683 non-null  int64  
1   artist_hotttnesss  193683 non-null  float64
2   song_year          193683 non-null  int64  
3   duration           193683 non-null  float64
dtypes: float64(2), int64(2)
memory usage: 7.4 MB
去重artist_hotttnesss为空，为0***********************
<class 'pandas.core.frame.DataFrame'>
Int64Index: 193683 entries, 0 to 2052373
Data columns (total 4 columns):
#   Column             Non-Null Count   Dtype
---  ------             --------------   -----  
0   song               193683 non-null  int64  
1   artist_hotttnesss  193683 non-null  float64
2   song_year          193683 non-null  int64  
3   duration           193683 non-null  float64
dtypes: float64(2), int64(2)
memory usage: 7.4 MB
duration***********************
<class 'pandas.core.frame.DataFrame'>
Int64Index: 193683 entries, 0 to 2052373
Data columns (total 4 columns):
#   Column             Non-Null Count   Dtype
---  ------             --------------   -----  
0   song               193683 non-null  int64  
1   artist_hotttnesss  193683 non-null  float64
2   song_year          193683 non-null  int64  
3   duration           193683 non-null  float64
dtypes: float64(2), int64(2)
memory usage: 7.4 MB
Duplicated rows in ratings file: 52374
Number of users: 41756
Number of songs: 193683
Sparsity: 0.025%
<class 'pandas.core.frame.DataFrame'>
Int64Index: 2052374 entries, 0 to 2052373
Data columns (total 6 columns):
#   Column             Non-Null Count    Dtype
---  ------             --------------    -----  
0   user               2052374 non-null  int64  
1   song               2052374 non-null  int64  
2   rating             2052374 non-null  float64
3   artist_hotttnesss  2052374 non-null  float64
4   song_year          2052374 non-null  int64  
5   duration           2052374 non-null  float64
dtypes: float64(3), int64(3)
memory usage: 109.6 MB
Number of users: 41756
Number of songs: 193683
Sparsity: 0.025%
Number of users: 41756
Number of songs: 193683
Sparsity: 0.025%
n_users: 41756, n_items: 108210, data sparsity: 0.0363 %
============================== DeepFM ==============================
2022-03-06 12:00:37.018394: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-03-06 12:00:37.020631: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'nvcuda.dll'; dlerror: nvcuda.dll not found
2022-03-06 12:00:37.020750: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2022-03-06 12:00:37.022557: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169] retrieving CUDA diagnostic information for host: ��˹��
2022-03-06 12:00:37.022690: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176] hostname: ��˹��
Training start time: 2022-03-06 12:00:37
total params: 2,703,399 | embedding params: 2,551,569 | network params: 151,830
train: 100%|██████████| 6416/6416 [01:31<00:00, 70.05it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 1 elapsed: 91.958s
train_loss: 0.0046
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 59.64it/s]
eval rmse: 0.0304
eval mae: 0.0150
eval r2: 0.2727
==============================
train: 100%|██████████| 6416/6416 [01:33<00:00, 68.41it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 2 elapsed: 94.218s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 68.00it/s]
eval rmse: 0.0302
eval mae: 0.0135
eval r2: 0.2834
==============================
train: 100%|██████████| 6416/6416 [01:31<00:00, 70.28it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 3 elapsed: 91.687s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 68.41it/s]
eval rmse: 0.0304
eval mae: 0.0149
eval r2: 0.2761
==============================
train: 100%|██████████| 6416/6416 [01:31<00:00, 69.98it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 4 elapsed: 92.075s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 65.38it/s]
eval rmse: 0.0305
eval mae: 0.0154
eval r2: 0.2712
==============================
train: 100%|██████████| 6416/6416 [01:31<00:00, 70.36it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 5 elapsed: 91.603s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 64.25it/s]
eval rmse: 0.0305
eval mae: 0.0150
eval r2: 0.2697
==============================
train: 100%|██████████| 6416/6416 [01:39<00:00, 64.34it/s]
Epoch 6 elapsed: 100.143s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 64.33it/s]
eval rmse: 0.0306
eval mae: 0.0154
eval r2: 0.2629
==============================
train: 100%|██████████| 6416/6416 [01:39<00:00, 64.30it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 7 elapsed: 100.199s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 62.63it/s]
eval rmse: 0.0306
eval mae: 0.0146
eval r2: 0.2631
==============================
train: 100%|██████████| 6416/6416 [01:33<00:00, 68.30it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 8 elapsed: 94.380s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 66.15it/s]
eval rmse: 0.0309
eval mae: 0.0157
eval r2: 0.2479
==============================
train: 100%|██████████| 6416/6416 [01:31<00:00, 70.21it/s]
Epoch 9 elapsed: 91.798s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 72.60it/s]
eval rmse: 0.0308
eval mae: 0.0144
eval r2: 0.2535
==============================
train: 100%|██████████| 6416/6416 [01:27<00:00, 73.26it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 10 elapsed: 87.957s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 68.41it/s]
eval rmse: 0.0331
eval mae: 0.0211
eval r2: 0.1400
==============================
train: 100%|██████████| 6416/6416 [01:28<00:00, 72.30it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 11 elapsed: 89.168s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 72.19it/s]
eval rmse: 0.0309
eval mae: 0.0143
eval r2: 0.2481
==============================
train: 100%|██████████| 6416/6416 [01:25<00:00, 74.89it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 12 elapsed: 86.051s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 75.03it/s]
eval rmse: 0.0316
eval mae: 0.0174
eval r2: 0.2144
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 76.29it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 13 elapsed: 84.509s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 74.29it/s]
eval rmse: 0.0319
eval mae: 0.0177
eval r2: 0.2024
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 76.37it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 14 elapsed: 84.417s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 72.54it/s]
eval rmse: 0.0314
eval mae: 0.0160
eval r2: 0.2248
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 76.07it/s]
Epoch 15 elapsed: 84.744s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 72.54it/s]
eval rmse: 0.0315
eval mae: 0.0151
eval r2: 0.2224
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 76.34it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 16 elapsed: 84.434s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 74.45it/s]
eval rmse: 0.0317
eval mae: 0.0150
eval r2: 0.2121
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 76.05it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 17 elapsed: 84.728s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 74.14it/s]
eval rmse: 0.0319
eval mae: 0.0167
eval r2: 0.2027
==============================
train: 100%|██████████| 6416/6416 [01:23<00:00, 76.65it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 18 elapsed: 84.052s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 73.91it/s]
eval rmse: 0.0316
eval mae: 0.0153
eval r2: 0.2136
==============================
train: 100%|██████████| 6416/6416 [01:23<00:00, 76.62it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 19 elapsed: 84.078s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 75.04it/s]
eval rmse: 0.0322
eval mae: 0.0163
eval r2: 0.1862
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 76.24it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 20 elapsed: 84.512s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 75.03it/s]
eval rmse: 0.0319
eval mae: 0.0153
eval r2: 0.2012
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 76.10it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 21 elapsed: 84.667s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 72.54it/s]
eval rmse: 0.0328
eval mae: 0.0181
eval r2: 0.1539
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 76.23it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 22 elapsed: 84.525s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 75.04it/s]
eval rmse: 0.0335
eval mae: 0.0181
eval r2: 0.1187
==============================
train: 100%|██████████| 6416/6416 [44:33<00:00,  2.40it/s]
Epoch 23 elapsed: 2674.123s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 70.12it/s]
eval rmse: 0.0335
eval mae: 0.0175
eval r2: 0.1197
==============================
train: 100%|██████████| 6416/6416 [01:23<00:00, 76.41it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 24 elapsed: 84.360s
train_loss: 0.0005
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 75.03it/s]
eval rmse: 0.0351
eval mae: 0.0184
eval r2: 0.0340
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 75.57it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 25 elapsed: 85.252s
train_loss: 0.0005
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 75.01it/s]
eval rmse: 0.0337
eval mae: 0.0171
eval r2: 0.1093
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 76.14it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 26 elapsed: 84.605s
train_loss: 0.0005
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 75.04it/s]
eval rmse: 0.0337
eval mae: 0.0180
eval r2: 0.1062
==============================
train: 100%|██████████| 6416/6416 [01:23<00:00, 76.59it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 27 elapsed: 84.137s
train_loss: 0.0005
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 71.98it/s]
eval rmse: 0.0344
eval mae: 0.0175
eval r2: 0.0686
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 76.37it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 28 elapsed: 84.365s
train_loss: 0.0004
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 75.04it/s]
eval rmse: 0.0343
eval mae: 0.0182
eval r2: 0.0731
==============================
train: 100%|██████████| 6416/6416 [01:23<00:00, 76.62it/s]
Epoch 29 elapsed: 84.081s
train_loss: 0.0004
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 72.54it/s]
eval rmse: 0.0342
eval mae: 0.0179
eval r2: 0.0825
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 75.83it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 30 elapsed: 84.962s
train_loss: 0.0004
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 70.33it/s]
eval rmse: 0.0338
eval mae: 0.0166
eval r2: 0.1015
==============================
train: 100%|██████████| 6416/6416 [01:34<00:00, 68.17it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 31 elapsed: 94.490s
train_loss: 0.0004
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 66.80it/s]
eval rmse: 0.0339
eval mae: 0.0166
eval r2: 0.0971
==============================
train: 100%|██████████| 6416/6416 [01:33<00:00, 68.55it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 32 elapsed: 94.018s
train_loss: 0.0004
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 66.15it/s]
eval rmse: 0.0335
eval mae: 0.0166
eval r2: 0.1195
==============================
train: 100%|██████████| 6416/6416 [01:34<00:00, 67.95it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 33 elapsed: 94.870s
train_loss: 0.0004
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 65.13it/s]
eval rmse: 0.0335
eval mae: 0.0168
eval r2: 0.1185
==============================
train: 100%|██████████| 6416/6416 [01:34<00:00, 67.96it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 34 elapsed: 94.835s
train_loss: 0.0003
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 66.93it/s]
eval rmse: 0.0343
eval mae: 0.0178
eval r2: 0.0742
==============================
train: 100%|██████████| 6416/6416 [01:32<00:00, 69.07it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 35 elapsed: 93.299s
train_loss: 0.0003
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 66.54it/s]
eval rmse: 0.0345
eval mae: 0.0181
eval r2: 0.0653
==============================
train: 100%|██████████| 6416/6416 [01:32<00:00, 69.11it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 36 elapsed: 93.228s
train_loss: 0.0003
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 66.80it/s]
eval rmse: 0.0344
eval mae: 0.0183
eval r2: 0.0679
==============================
train: 100%|██████████| 6416/6416 [01:32<00:00, 69.32it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 37 elapsed: 92.939s
train_loss: 0.0003
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 66.67it/s]
eval rmse: 0.0350
eval mae: 0.0189
eval r2: 0.0382
==============================
train: 100%|██████████| 6416/6416 [01:32<00:00, 69.45it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 38 elapsed: 92.811s
train_loss: 0.0003
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 66.02it/s]
eval rmse: 0.0347
eval mae: 0.0190
eval r2: 0.0543
==============================
train: 100%|██████████| 6416/6416 [01:32<00:00, 69.09it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 39 elapsed: 93.264s
train_loss: 0.0003
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 66.80it/s]
eval rmse: 0.0345
eval mae: 0.0176
eval r2: 0.0653
==============================
train: 100%|██████████| 6416/6416 [01:32<00:00, 69.01it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 40 elapsed: 93.385s
train_loss: 0.0003
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 66.80it/s]
eval rmse: 0.0348
eval mae: 0.0178
eval r2: 0.0494
==============================
train: 100%|██████████| 6416/6416 [01:32<00:00, 69.01it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 41 elapsed: 93.412s
train_loss: 0.0003
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 67.33it/s]
eval rmse: 0.0349
eval mae: 0.0184
eval r2: 0.0452
==============================
train: 100%|██████████| 6416/6416 [01:28<00:00, 72.71it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 42 elapsed: 88.649s
train_loss: 0.0003
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 74.80it/s]
eval rmse: 0.0394
eval mae: 0.0230
eval r2: -0.2174
==============================
train: 100%|██████████| 6416/6416 [01:25<00:00, 75.40it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 43 elapsed: 85.453s
train_loss: 0.0003
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 70.19it/s]
eval rmse: 0.0348
eval mae: 0.0177
eval r2: 0.0476
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 75.93it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 44 elapsed: 84.854s
train_loss: 0.0002
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 72.54it/s]
eval rmse: 0.0354
eval mae: 0.0198
eval r2: 0.0179
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 76.25it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 45 elapsed: 84.497s
train_loss: 0.0002
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 68.00it/s]
eval rmse: 0.0342
eval mae: 0.0174
eval r2: 0.0795
==============================
train: 100%|██████████| 6416/6416 [01:26<00:00, 74.25it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 46 elapsed: 86.807s
train_loss: 0.0002
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 70.20it/s]
eval rmse: 0.0342
eval mae: 0.0170
eval r2: 0.0824
==============================
train: 100%|██████████| 6416/6416 [01:25<00:00, 75.23it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 47 elapsed: 85.649s
train_loss: 0.0002
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 62.96it/s]
eval rmse: 0.0371
eval mae: 0.0197
eval r2: -0.0803
==============================
train: 100%|██████████| 6416/6416 [01:26<00:00, 74.58it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 48 elapsed: 86.405s
train_loss: 0.0002
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 74.94it/s]
eval rmse: 0.0348
eval mae: 0.0179
eval r2: 0.0501
==============================
train: 100%|██████████| 6416/6416 [01:26<00:00, 74.30it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 49 elapsed: 86.685s
train_loss: 0.0002
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 63.31it/s]
eval rmse: 0.0352
eval mae: 0.0186
eval r2: 0.0282
==============================
train: 100%|██████████| 6416/6416 [01:27<00:00, 73.73it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 50 elapsed: 87.459s
train_loss: 0.0002
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 75.03it/s]
eval rmse: 0.0349
eval mae: 0.0176
eval r2: 0.0416
==============================
train: 100%|██████████| 6416/6416 [01:29<00:00, 71.37it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 51 elapsed: 90.239s
train_loss: 0.0002
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 72.55it/s]
eval rmse: 0.0349
eval mae: 0.0182
eval r2: 0.0420
==============================
train: 100%|██████████| 6416/6416 [01:34<00:00, 68.01it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 52 elapsed: 94.729s
train_loss: 0.0002
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 58.81it/s]
eval rmse: 0.0343
eval mae: 0.0171
eval r2: 0.0728
==============================
train: 100%|██████████| 6416/6416 [01:31<00:00, 70.37it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 53 elapsed: 91.595s
train_loss: 0.0002
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 64.00it/s]
eval rmse: 0.0347
eval mae: 0.0182
eval r2: 0.0519
==============================
train: 100%|██████████| 6416/6416 [01:31<00:00, 70.01it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 54 elapsed: 92.041s
train_loss: 0.0002
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 70.08it/s]
eval rmse: 0.0342
eval mae: 0.0174
eval r2: 0.0818
==============================
train: 100%|██████████| 6416/6416 [01:27<00:00, 73.73it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 55 elapsed: 87.389s
train_loss: 0.0002
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 64.01it/s]
eval rmse: 0.0350
eval mae: 0.0184
eval r2: 0.0390
==============================
train: 100%|██████████| 6416/6416 [01:29<00:00, 71.49it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 56 elapsed: 90.119s
train_loss: 0.0002
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 69.52it/s]
eval rmse: 0.0339
eval mae: 0.0164
eval r2: 0.0949
==============================
train: 100%|██████████| 6416/6416 [01:26<00:00, 74.54it/s]
Epoch 57 elapsed: 86.458s
train_loss: 0.0002
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 70.20it/s]
eval rmse: 0.0350
eval mae: 0.0181
eval r2: 0.0387
==============================
train: 100%|██████████| 6416/6416 [01:26<00:00, 73.97it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 58 elapsed: 87.125s
train_loss: 0.0002
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 72.54it/s]
eval rmse: 0.0344
eval mae: 0.0175
eval r2: 0.0676
==============================
train: 100%|██████████| 6416/6416 [01:31<00:00, 70.25it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 59 elapsed: 91.703s
train_loss: 0.0002
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 60.44it/s]
eval rmse: 0.0358
eval mae: 0.0185
eval r2: -0.0072
==============================
train: 100%|██████████| 6416/6416 [01:27<00:00, 73.39it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 60 elapsed: 87.826s
train_loss: 0.0002
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 72.54it/s]
eval rmse: 0.0344
eval mae: 0.0172
eval r2: 0.0702
==============================
train: 100%|██████████| 6416/6416 [01:25<00:00, 74.71it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 61 elapsed: 86.239s
train_loss: 0.0002
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 71.00it/s]
eval rmse: 0.0352
eval mae: 0.0184
eval r2: 0.0280
==============================
train: 100%|██████████| 6416/6416 [01:26<00:00, 74.25it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 62 elapsed: 86.783s
train_loss: 0.0002
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 69.80it/s]
eval rmse: 0.0342
eval mae: 0.0167
eval r2: 0.0810
==============================
train: 100%|██████████| 6416/6416 [01:26<00:00, 74.37it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 63 elapsed: 86.624s
train_loss: 0.0002
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 75.03it/s]
eval rmse: 0.0353
eval mae: 0.0183
eval r2: 0.0200
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 75.51it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 64 elapsed: 85.329s
train_loss: 0.0002
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 72.53it/s]
eval rmse: 0.0348
eval mae: 0.0179
eval r2: 0.0485
==============================
train: 100%|██████████| 6416/6416 [01:27<00:00, 73.05it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 65 elapsed: 88.178s
train_loss: 0.0002
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 74.05it/s]
eval rmse: 0.0343
eval mae: 0.0175
eval r2: 0.0752
==============================
train: 100%|██████████| 6416/6416 [01:25<00:00, 75.03it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 66 elapsed: 85.863s
train_loss: 0.0001
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 72.36it/s]
eval rmse: 0.0349
eval mae: 0.0185
eval r2: 0.0414
==============================
train: 100%|██████████| 6416/6416 [01:25<00:00, 74.96it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 67 elapsed: 85.950s
train_loss: 0.0001
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 70.94it/s]
eval rmse: 0.0350
eval mae: 0.0180
eval r2: 0.0374
==============================
train: 100%|██████████| 6416/6416 [01:26<00:00, 74.42it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 68 elapsed: 86.578s
train_loss: 0.0001
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 75.03it/s]
eval rmse: 0.0358
eval mae: 0.0193
eval r2: -0.0069
==============================
train: 100%|██████████| 6416/6416 [01:25<00:00, 74.64it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 69 elapsed: 86.306s
train_loss: 0.0001
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 70.82it/s]
eval rmse: 0.0348
eval mae: 0.0174
eval r2: 0.0456
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 76.12it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 70 elapsed: 84.638s
train_loss: 0.0001
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 72.49it/s]
eval rmse: 0.0345
eval mae: 0.0172
eval r2: 0.0635
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 76.34it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 71 elapsed: 84.399s
train_loss: 0.0001
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 75.04it/s]
eval rmse: 0.0340
eval mae: 0.0166
eval r2: 0.0891
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 76.13it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 72 elapsed: 84.644s
train_loss: 0.0001
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 74.13it/s]
eval rmse: 0.0346
eval mae: 0.0172
eval r2: 0.0586
==============================
train: 100%|██████████| 6416/6416 [01:23<00:00, 76.39it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 73 elapsed: 84.350s
train_loss: 0.0001
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 72.53it/s]
eval rmse: 0.0344
eval mae: 0.0172
eval r2: 0.0699
==============================
train: 100%|██████████| 6416/6416 [01:23<00:00, 76.38it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 74 elapsed: 84.348s
train_loss: 0.0001
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 75.04it/s]
eval rmse: 0.0348
eval mae: 0.0177
eval r2: 0.0501
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 76.33it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 75 elapsed: 84.421s
train_loss: 0.0001
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 71.74it/s]
eval rmse: 0.0356
eval mae: 0.0187
eval r2: 0.0050
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 75.90it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 76 elapsed: 84.890s
train_loss: 0.0001
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 72.54it/s]
eval rmse: 0.0347
eval mae: 0.0180
eval r2: 0.0555
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 75.73it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 77 elapsed: 85.056s
train_loss: 0.0001
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 64.86it/s]
eval rmse: 0.0351
eval mae: 0.0181
eval r2: 0.0316
==============================
train: 100%|██████████| 6416/6416 [01:25<00:00, 75.24it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 78 elapsed: 85.624s
train_loss: 0.0001
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 72.50it/s]
eval rmse: 0.0351
eval mae: 0.0176
eval r2: 0.0341
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 76.06it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 79 elapsed: 84.713s
train_loss: 0.0001
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 72.42it/s]
eval rmse: 0.0343
eval mae: 0.0169
eval r2: 0.0767
==============================
train: 100%|██████████| 6416/6416 [01:25<00:00, 75.25it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 80 elapsed: 85.615s
train_loss: 0.0001
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 74.02it/s]
eval rmse: 0.0347
eval mae: 0.0178
eval r2: 0.0552
==============================
*******************************************结束！*******************************************
