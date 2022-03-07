# user  song都筛选后

D:\Develop\Python37\python.exe C:\Users\73556\AppData\Roaming\JetBrains\IntelliJIdea2021.2\plugins\python\helpers\pydev\pydevconsole.py --mode=client --port=65447
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\IdeaSpace\\PythonMusicRecommend', 'D:/IdeaSpace/PythonMusicRecommend'])
PyDev console: starting.
Python 3.7.2 (tags/v3.7.2:9a3ffc0492, Dec 23 2018, 23:09:28) [MSC v.1916 64 bit (AMD64)] on win32
runfile('D:/IdeaSpace/PythonMusicRecommend/LibRecommender-master/test/librec_feat.py', wdir='D:/IdeaSpace/PythonMusicRecommend/LibRecommender-master/test')
2022-03-06 17:54:17.730916: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-03-06 17:54:17.731053: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
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
Duplicated rows in ratings file: 37143
Number of users: 16834
Number of songs: 21213
Sparsity: 0.274%
<class 'pandas.core.frame.DataFrame'>
Int64Index: 979376 entries, 0 to 979375
Data columns (total 6 columns):
#   Column             Non-Null Count   Dtype
---  ------             --------------   -----  
0   user               979376 non-null  int64  
1   song               979376 non-null  int64  
2   rating             979376 non-null  float64
3   artist_hotttnesss  979376 non-null  float64
4   song_year          979376 non-null  int64  
5   duration           979376 non-null  float64
dtypes: float64(3), int64(3)
memory usage: 52.3 MB
Number of users: 16834
Number of songs: 21213
Sparsity: 0.274%
Number of users: 16834
Number of songs: 21213
Sparsity: 0.274%
n_users: 16834, n_items: 17255, data sparsity: 0.2697 %
============================== Wide_Deep ==============================
Training start time: 2022-03-06 17:54:49
2022-03-06 17:54:49.124531: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-03-06 17:54:49.127042: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'nvcuda.dll'; dlerror: nvcuda.dll not found
2022-03-06 17:54:49.127156: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2022-03-06 17:54:49.128805: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169] retrieving CUDA diagnostic information for host: ��˹��
2022-03-06 17:54:49.128918: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176] hostname: ��˹��
total params: 733,201 | embedding params: 581,388 | network params: 151,813
train: 100%|██████████| 3061/3061 [00:11<00:00, 262.93it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 1 elapsed: 11.790s
train_loss: 0.0111
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 76.80it/s]
eval rmse: 0.0252
eval mae: 0.0124
eval r2: 0.1110
==============================
train: 100%|██████████| 3061/3061 [00:11<00:00, 259.76it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 2 elapsed: 11.932s
train_loss: 0.0005
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 82.28it/s]
eval rmse: 0.0253
eval mae: 0.0111
eval r2: 0.1060
==============================
train: 100%|██████████| 3061/3061 [00:11<00:00, 261.84it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 3 elapsed: 11.854s
train_loss: 0.0005
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 88.54it/s]
eval rmse: 0.0256
eval mae: 0.0113
eval r2: 0.0851
==============================
train: 100%|██████████| 3061/3061 [00:11<00:00, 265.06it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 4 elapsed: 11.684s
train_loss: 0.009
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 88.61it/s]
eval rmse: 0.0251
eval mae: 0.0110
eval r2: 0.1161
==============================
train: 100%|██████████| 3061/3061 [00:11<00:00, 270.91it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 5 elapsed: 11.461s
train_loss: 0.0004
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 88.62it/s]
eval rmse: 0.0254
eval mae: 0.0108
eval r2: 0.0947
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 241.64it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 6 elapsed: 12.819s
train_loss: 0.0004
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 80.98it/s]
eval rmse: 0.0261
eval mae: 0.0115
eval r2: 0.0460
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 240.18it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 7 elapsed: 12.928s
train_loss: 0.0004
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 89.99it/s]
eval rmse: 0.0262
eval mae: 0.0119
eval r2: 0.0419
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 252.34it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 8 elapsed: 12.275s
train_loss: 0.0004
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 88.62it/s]
eval rmse: 0.0283
eval mae: 0.0132
eval r2: -0.1205
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 252.41it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 9 elapsed: 12.288s
train_loss: 0.0004
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 88.61it/s]
eval rmse: 0.0265
eval mae: 0.0115
eval r2: 0.0175
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 241.97it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 10 elapsed: 12.825s
train_loss: 0.0004
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 82.28it/s]
eval rmse: 0.0280
eval mae: 0.0118
eval r2: -0.0973
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 246.95it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 11 elapsed: 12.546s
train_loss: 0.0003
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 88.62it/s]
eval rmse: 0.0289
eval mae: 0.0134
eval r2: -0.1686
==============================
train: 100%|██████████| 3061/3061 [00:11<00:00, 257.98it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 12 elapsed: 12.005s
train_loss: 0.0003
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 95.28it/s]
eval rmse: 0.0287
eval mae: 0.0123
eval r2: -0.1489
==============================
train: 100%|██████████| 3061/3061 [00:11<00:00, 263.81it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 13 elapsed: 11.769s
train_loss: 0.0003
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 86.64it/s]
eval rmse: 0.0291
eval mae: 0.0156
eval r2: -0.1811
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 246.98it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 14 elapsed: 12.540s
train_loss: 0.0003
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 88.62it/s]
eval rmse: 0.0306
eval mae: 0.0132
eval r2: -0.3065
==============================
train: 100%|██████████| 3061/3061 [00:11<00:00, 256.60it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 15 elapsed: 12.070s
train_loss: 0.0003
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 78.60it/s]
eval rmse: 0.0296
eval mae: 0.0120
eval r2: -0.2215
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 239.58it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 16 elapsed: 12.979s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 82.29it/s]
eval rmse: 0.0297
eval mae: 0.0120
eval r2: -0.2348
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 244.67it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 17 elapsed: 12.679s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 82.18it/s]
eval rmse: 0.0309
eval mae: 0.0139
eval r2: -0.3371
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 242.75it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 18 elapsed: 12.759s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 77.59it/s]
eval rmse: 0.0301
eval mae: 0.0121
eval r2: -0.2634
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 242.09it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 19 elapsed: 12.856s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 82.29it/s]
eval rmse: 0.0312
eval mae: 0.0130
eval r2: -0.3650
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 246.28it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 20 elapsed: 12.579s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 88.62it/s]
eval rmse: 0.0310
eval mae: 0.0133
eval r2: -0.3442
==============================
train: 100%|██████████| 3061/3061 [00:11<00:00, 258.77it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 21 elapsed: 11.990s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 88.61it/s]
eval rmse: 0.0304
eval mae: 0.0122
eval r2: -0.2918
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 253.45it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 22 elapsed: 12.243s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 80.57it/s]
eval rmse: 0.0326
eval mae: 0.0139
eval r2: -0.4897
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 244.65it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 23 elapsed: 12.683s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 82.29it/s]
eval rmse: 0.0295
eval mae: 0.0127
eval r2: -0.2191
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 246.88it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 24 elapsed: 12.576s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 88.61it/s]
eval rmse: 0.0306
eval mae: 0.0124
eval r2: -0.3124
==============================
train: 100%|██████████| 3061/3061 [00:11<00:00, 259.42it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 25 elapsed: 11.950s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 84.72it/s]
eval rmse: 0.0318
eval mae: 0.0140
eval r2: -0.4126
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 251.11it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 26 elapsed: 12.351s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 82.29it/s]
eval rmse: 0.0323
eval mae: 0.0136
eval r2: -0.4621
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 245.83it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 27 elapsed: 12.622s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 82.29it/s]
eval rmse: 0.0304
eval mae: 0.0121
eval r2: -0.2930
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 240.80it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 28 elapsed: 12.880s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 82.28it/s]
eval rmse: 0.0308
eval mae: 0.0126
eval r2: -0.3258
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 242.95it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 29 elapsed: 12.764s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 82.29it/s]
eval rmse: 0.0308
eval mae: 0.0126
eval r2: -0.3299
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 237.42it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 30 elapsed: 13.049s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 82.28it/s]
eval rmse: 0.0311
eval mae: 0.0128
eval r2: -0.3516
==============================
train: 100%|██████████| 3061/3061 [00:13<00:00, 230.16it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 31 elapsed: 13.474s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 82.28it/s]
eval rmse: 0.0318
eval mae: 0.0131
eval r2: -0.4119
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 244.21it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 32 elapsed: 12.701s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 88.61it/s]
eval rmse: 0.0301
eval mae: 0.0125
eval r2: -0.2683
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 243.59it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 33 elapsed: 12.721s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 88.40it/s]
eval rmse: 0.0303
eval mae: 0.0124
eval r2: -0.2813
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 248.73it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 34 elapsed: 12.470s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 76.92it/s]
eval rmse: 0.0301
eval mae: 0.0130
eval r2: -0.2630
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 247.19it/s]
Epoch 35 elapsed: 12.608s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 87.70it/s]
eval rmse: 0.0308
eval mae: 0.0127
eval r2: -0.3284
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 250.86it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 36 elapsed: 12.359s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 88.62it/s]
eval rmse: 0.0307
eval mae: 0.0127
eval r2: -0.3184
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 251.29it/s]
Epoch 37 elapsed: 12.366s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 76.80it/s]
eval rmse: 0.0310
eval mae: 0.0132
eval r2: -0.3466
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 242.04it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 38 elapsed: 12.824s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 81.08it/s]
eval rmse: 0.0307
eval mae: 0.0127
eval r2: -0.3152
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 237.00it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 39 elapsed: 13.099s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 82.29it/s]
eval rmse: 0.0311
eval mae: 0.0130
eval r2: -0.3560
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 241.24it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 40 elapsed: 12.848s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 82.28it/s]
eval rmse: 0.0293
eval mae: 0.0121
eval r2: -0.2030
==============================
train: 100%|██████████| 3061/3061 [00:13<00:00, 234.33it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 41 elapsed: 13.204s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 82.29it/s]
eval rmse: 0.0300
eval mae: 0.0124
eval r2: -0.2622
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 245.22it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 42 elapsed: 12.630s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 88.62it/s]
eval rmse: 0.0312
eval mae: 0.0129
eval r2: -0.3586
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 243.64it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 43 elapsed: 12.717s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 82.29it/s]
eval rmse: 0.0304
eval mae: 0.0129
eval r2: -0.2895
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 240.05it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 44 elapsed: 12.897s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 88.62it/s]
eval rmse: 0.0304
eval mae: 0.0125
eval r2: -0.2966
==============================
train: 100%|██████████| 3061/3061 [00:11<00:00, 255.35it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 45 elapsed: 12.138s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 76.80it/s]
eval rmse: 0.0303
eval mae: 0.0126
eval r2: -0.2857
==============================
train: 100%|██████████| 3061/3061 [00:12<00:00, 235.70it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 46 elapsed: 13.150s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 64.00it/s]
eval rmse: 0.0300
eval mae: 0.0124
eval r2: -0.2585
==============================
train: 100%|██████████| 3061/3061 [00:14<00:00, 211.52it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 47 elapsed: 14.628s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 67.77it/s]
eval rmse: 0.0305
eval mae: 0.0127
eval r2: -0.3002
==============================
train: 100%|██████████| 3061/3061 [00:13<00:00, 220.01it/s]
Epoch 48 elapsed: 14.069s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 73.42it/s]
eval rmse: 0.0300
eval mae: 0.0133
eval r2: -0.2596
==============================
train: 100%|██████████| 3061/3061 [00:13<00:00, 227.53it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 49 elapsed: 13.616s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 72.00it/s]
eval rmse: 0.0301
eval mae: 0.0123
eval r2: -0.2690
==============================
train: 100%|██████████| 3061/3061 [00:14<00:00, 218.12it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 50 elapsed: 14.209s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 76.80it/s]
eval rmse: 0.0301
eval mae: 0.0123
eval r2: -0.2661
==============================
*******************************************结束！*******************************************
