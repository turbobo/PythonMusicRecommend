D:\Develop\Python37\python.exe C:\Users\73556\AppData\Roaming\JetBrains\IntelliJIdea2021.2\plugins\python\helpers\pydev\pydevconsole.py --mode=client --port=50543
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\IdeaSpace\\PythonMusicRecommend', 'D:/IdeaSpace/PythonMusicRecommend'])
PyDev console: starting.
Python 3.7.2 (tags/v3.7.2:9a3ffc0492, Dec 23 2018, 23:09:28) [MSC v.1916 64 bit (AMD64)] on win32
runfile('D:/IdeaSpace/PythonMusicRecommend/LibRecommender-master/test/librec_feat.py', wdir='D:/IdeaSpace/PythonMusicRecommend/LibRecommender-master/test')
2022-03-06 20:05:51.266315: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-03-06 20:05:51.266462: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
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
============================== DeepFM ==============================
Training start time: 2022-03-06 20:06:23
2022-03-06 20:06:23.882601: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-03-06 20:06:23.884718: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'nvcuda.dll'; dlerror: nvcuda.dll not found
2022-03-06 20:06:23.884851: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2022-03-06 20:06:23.886452: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169] retrieving CUDA diagnostic information for host: ��˹��
2022-03-06 20:06:23.886610: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176] hostname: ��˹��
total params: 733,218 | embedding params: 581,388 | network params: 151,830
train: 100%|██████████| 3061/3061 [00:19<00:00, 159.25it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 1 elapsed: 19.393s
train_loss: 0.0086
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 57.88it/s]
eval rmse: 0.0253
eval mae: 0.0115
eval r2: 0.1079
==============================
train: 100%|██████████| 3061/3061 [00:18<00:00, 167.49it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 2 elapsed: 18.476s
train_loss: 0.0008
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 67.67it/s]
eval rmse: 0.0252
eval mae: 0.0104
eval r2: 0.1143
==============================
train: 100%|██████████| 3061/3061 [00:17<00:00, 174.67it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 3 elapsed: 17.709s
train_loss: 0.0006
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 67.42it/s]
eval rmse: 0.0250
eval mae: 0.0116
eval r2: 0.1292
==============================
train: 100%|██████████| 3061/3061 [00:17<00:00, 173.54it/s]
Epoch 4 elapsed: 17.818s
train_loss: 0.0005
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 68.44it/s]
eval rmse: 0.0250
eval mae: 0.0109
eval r2: 0.1284
==============================
train: 100%|██████████| 3061/3061 [00:18<00:00, 167.30it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 5 elapsed: 18.469s
train_loss: 0.0005
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 60.81it/s]
eval rmse: 0.0250
eval mae: 0.0113
eval r2: 0.1252
==============================
train: 100%|██████████| 3061/3061 [00:18<00:00, 169.55it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 6 elapsed: 18.237s
train_loss: 0.0005
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 65.45it/s]
eval rmse: 0.0252
eval mae: 0.0115
eval r2: 0.1146
==============================
train: 100%|██████████| 3061/3061 [00:18<00:00, 168.78it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 7 elapsed: 18.317s
train_loss: 0.0005
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 67.79it/s]
eval rmse: 0.0252
eval mae: 0.0111
eval r2: 0.1127
==============================
train: 100%|██████████| 3061/3061 [00:19<00:00, 159.94it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 8 elapsed: 19.323s
train_loss: 0.0005
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 65.22it/s]
eval rmse: 0.0257
eval mae: 0.0125
eval r2: 0.0779
==============================
train: 100%|██████████| 3061/3061 [00:17<00:00, 170.42it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 9 elapsed: 18.152s
train_loss: 0.0005
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 67.67it/s]
eval rmse: 0.0254
eval mae: 0.0118
eval r2: 0.0944
==============================
train: 100%|██████████| 3061/3061 [00:18<00:00, 169.61it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 10 elapsed: 18.219s
train_loss: 0.0005
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 63.60it/s]
eval rmse: 0.0257
eval mae: 0.0124
eval r2: 0.0768
==============================
train: 100%|██████████| 3061/3061 [00:18<00:00, 168.38it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 11 elapsed: 18.372s
train_loss: 0.0004
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 62.94it/s]
eval rmse: 0.0259
eval mae: 0.0120
eval r2: 0.0623
==============================
train: 100%|██████████| 3061/3061 [00:18<00:00, 166.42it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 12 elapsed: 18.562s
train_loss: 0.0004
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 65.46it/s]
eval rmse: 0.0260
eval mae: 0.0119
eval r2: 0.0564
==============================
train: 100%|██████████| 3061/3061 [00:18<00:00, 169.36it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 13 elapsed: 18.262s
train_loss: 0.0004
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 65.81it/s]
eval rmse: 0.0266
eval mae: 0.0130
eval r2: 0.0083
==============================
train: 100%|██████████| 3061/3061 [00:18<00:00, 169.50it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 14 elapsed: 18.247s
train_loss: 0.0004
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 66.91it/s]
eval rmse: 0.0262
eval mae: 0.0124
eval r2: 0.0419
==============================
train: 100%|██████████| 3061/3061 [00:18<00:00, 168.75it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 15 elapsed: 18.323s
train_loss: 0.0004
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 64.06it/s]
eval rmse: 0.0270
eval mae: 0.0132
eval r2: -0.0173
==============================
train: 100%|██████████| 3061/3061 [00:20<00:00, 151.82it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 16 elapsed: 20.342s
train_loss: 0.0004
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 64.29it/s]
eval rmse: 0.0264
eval mae: 0.0125
eval r2: 0.0287
==============================
train: 100%|██████████| 3061/3061 [00:18<00:00, 164.19it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 17 elapsed: 18.837s
train_loss: 0.0004
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 62.72it/s]
eval rmse: 0.0267
eval mae: 0.0128
eval r2: 0.0001
==============================
train: 100%|██████████| 3061/3061 [00:17<00:00, 173.29it/s]
Epoch 18 elapsed: 17.849s
train_loss: 0.0003
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 66.67it/s]
eval rmse: 0.0282
eval mae: 0.0148
eval r2: -0.1109
==============================
train: 100%|██████████| 3061/3061 [00:18<00:00, 168.93it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 19 elapsed: 18.304s
train_loss: 0.0003
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 62.28it/s]
eval rmse: 0.0279
eval mae: 0.0142
eval r2: -0.0856
==============================
train: 100%|██████████| 3061/3061 [00:18<00:00, 162.57it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 20 elapsed: 19.010s
train_loss: 0.0003
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 67.42it/s]
eval rmse: 0.0277
eval mae: 0.0136
eval r2: -0.0693
==============================
train: 100%|██████████| 3061/3061 [00:19<00:00, 159.94it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 21 elapsed: 19.317s
train_loss: 0.0003
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 59.21it/s]
eval rmse: 0.0275
eval mae: 0.0133
eval r2: -0.0602
==============================
train: 100%|██████████| 3061/3061 [00:20<00:00, 151.29it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 22 elapsed: 20.412s
train_loss: 0.0003
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 61.22it/s]
eval rmse: 0.0283
eval mae: 0.0139
eval r2: -0.1230
==============================
train: 100%|██████████| 3061/3061 [00:20<00:00, 146.51it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 23 elapsed: 21.069s
train_loss: 0.0003
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 62.72it/s]
eval rmse: 0.0271
eval mae: 0.0126
eval r2: -0.0287
==============================
train: 100%|██████████| 3061/3061 [00:21<00:00, 143.72it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 24 elapsed: 21.475s
train_loss: 0.0003
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 58.82it/s]
eval rmse: 0.0273
eval mae: 0.0134
eval r2: -0.0392
==============================
train: 100%|██████████| 3061/3061 [00:21<00:00, 145.48it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 25 elapsed: 21.222s
train_loss: 0.0003
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 63.38it/s]
eval rmse: 0.0276
eval mae: 0.0131
eval r2: -0.0678
==============================
train: 100%|██████████| 3061/3061 [00:20<00:00, 146.33it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 26 elapsed: 21.102s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 57.14it/s]
eval rmse: 0.0278
eval mae: 0.0139
eval r2: -0.0809
==============================
train: 100%|██████████| 3061/3061 [00:20<00:00, 146.10it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 27 elapsed: 21.132s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 63.61it/s]
eval rmse: 0.0281
eval mae: 0.0133
eval r2: -0.1005
==============================
train: 100%|██████████| 3061/3061 [00:21<00:00, 141.88it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 28 elapsed: 21.769s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 62.17it/s]
eval rmse: 0.0273
eval mae: 0.0128
eval r2: -0.0455
==============================
train: 100%|██████████| 3061/3061 [00:20<00:00, 151.57it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 29 elapsed: 20.382s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 63.60it/s]
eval rmse: 0.0287
eval mae: 0.0152
eval r2: -0.1532
==============================
train: 100%|██████████| 3061/3061 [00:18<00:00, 164.57it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 30 elapsed: 18.770s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 67.31it/s]
eval rmse: 0.0280
eval mae: 0.0138
eval r2: -0.0984
==============================
train: 100%|██████████| 3061/3061 [00:20<00:00, 149.40it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 31 elapsed: 20.662s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 60.20it/s]
eval rmse: 0.0287
eval mae: 0.0155
eval r2: -0.1528
==============================
train: 100%|██████████| 3061/3061 [00:21<00:00, 141.63it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 32 elapsed: 21.796s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 62.50it/s]
eval rmse: 0.0275
eval mae: 0.0129
eval r2: -0.0543
==============================
train: 100%|██████████| 3061/3061 [00:21<00:00, 143.88it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 33 elapsed: 21.454s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 61.85it/s]
eval rmse: 0.0277
eval mae: 0.0129
eval r2: -0.0731
==============================
train: 100%|██████████| 3061/3061 [00:18<00:00, 163.85it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 34 elapsed: 18.861s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 63.38it/s]
eval rmse: 0.0280
eval mae: 0.0142
eval r2: -0.0981
==============================
train: 100%|██████████| 3061/3061 [00:17<00:00, 178.92it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 35 elapsed: 17.280s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 66.42it/s]
eval rmse: 0.0278
eval mae: 0.0129
eval r2: -0.0807
==============================
train: 100%|██████████| 3061/3061 [00:17<00:00, 175.61it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 36 elapsed: 17.602s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 62.07it/s]
eval rmse: 0.0279
eval mae: 0.0140
eval r2: -0.0904
==============================
train: 100%|██████████| 3061/3061 [00:16<00:00, 185.33it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 37 elapsed: 16.681s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 73.62it/s]
eval rmse: 0.0285
eval mae: 0.0147
eval r2: -0.1398
==============================
train: 100%|██████████| 3061/3061 [00:16<00:00, 187.66it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 38 elapsed: 16.484s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 74.91it/s]
eval rmse: 0.0273
eval mae: 0.0128
eval r2: -0.0453
==============================
train: 100%|██████████| 3061/3061 [00:15<00:00, 194.42it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 39 elapsed: 15.896s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 75.22it/s]
eval rmse: 0.0285
eval mae: 0.0139
eval r2: -0.1396
==============================
train: 100%|██████████| 3061/3061 [00:16<00:00, 186.89it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 40 elapsed: 16.535s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 73.10it/s]
eval rmse: 0.0275
eval mae: 0.0128
eval r2: -0.0561
==============================
train: 100%|██████████| 3061/3061 [00:18<00:00, 166.61it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 41 elapsed: 18.520s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 63.38it/s]
eval rmse: 0.0277
eval mae: 0.0130
eval r2: -0.0770
==============================
train: 100%|██████████| 3061/3061 [00:19<00:00, 154.48it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 42 elapsed: 19.994s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 57.51it/s]
eval rmse: 0.0279
eval mae: 0.0135
eval r2: -0.0873
==============================
train: 100%|██████████| 3061/3061 [00:20<00:00, 147.35it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 43 elapsed: 20.963s
train_loss: 0.0002
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 62.07it/s]
eval rmse: 0.0281
eval mae: 0.0137
eval r2: -0.1012
==============================
train: 100%|██████████| 3061/3061 [00:21<00:00, 144.70it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 44 elapsed: 21.339s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 63.60it/s]
eval rmse: 0.0279
eval mae: 0.0131
eval r2: -0.0917
==============================
train: 100%|██████████| 3061/3061 [00:20<00:00, 151.09it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 45 elapsed: 20.448s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 56.96it/s]
eval rmse: 0.0277
eval mae: 0.0130
eval r2: -0.0764
==============================
train: 100%|██████████| 3061/3061 [00:21<00:00, 144.93it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 46 elapsed: 21.301s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 63.16it/s]
eval rmse: 0.0278
eval mae: 0.0131
eval r2: -0.0844
==============================
train: 100%|██████████| 3061/3061 [00:21<00:00, 145.21it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 47 elapsed: 21.257s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 56.78it/s]
eval rmse: 0.0276
eval mae: 0.0124
eval r2: -0.0622
==============================
train: 100%|██████████| 3061/3061 [00:19<00:00, 156.64it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 48 elapsed: 19.725s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 61.64it/s]
eval rmse: 0.0281
eval mae: 0.0132
eval r2: -0.1036
==============================
train: 100%|██████████| 3061/3061 [00:19<00:00, 155.09it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 49 elapsed: 19.920s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 62.94it/s]
eval rmse: 0.0276
eval mae: 0.0127
eval r2: -0.0669
==============================
train: 100%|██████████| 3061/3061 [00:17<00:00, 177.27it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 50 elapsed: 17.454s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 76.80it/s]
eval rmse: 0.0276
eval mae: 0.0128
eval r2: -0.0618
==============================
train: 100%|██████████| 3061/3061 [00:15<00:00, 192.76it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 51 elapsed: 16.044s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 72.01it/s]
eval rmse: 0.0282
eval mae: 0.0139
eval r2: -0.1132
==============================
train: 100%|██████████| 3061/3061 [00:16<00:00, 191.05it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 52 elapsed: 16.168s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 76.80it/s]
eval rmse: 0.0282
eval mae: 0.0134
eval r2: -0.1131
==============================
train: 100%|██████████| 3061/3061 [00:15<00:00, 201.65it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 53 elapsed: 15.332s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 75.24it/s]
eval rmse: 0.0283
eval mae: 0.0134
eval r2: -0.1207
==============================
train: 100%|██████████| 3061/3061 [00:15<00:00, 193.03it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 54 elapsed: 16.000s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 68.81it/s]
eval rmse: 0.0278
eval mae: 0.0133
eval r2: -0.0813
==============================
train: 100%|██████████| 3061/3061 [00:15<00:00, 195.96it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 55 elapsed: 15.788s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 70.90it/s]
eval rmse: 0.0283
eval mae: 0.0134
eval r2: -0.1228
==============================
train: 100%|██████████| 3061/3061 [00:16<00:00, 189.90it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 56 elapsed: 16.297s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 72.00it/s]
eval rmse: 0.0284
eval mae: 0.0144
eval r2: -0.1266
==============================
train: 100%|██████████| 3061/3061 [00:15<00:00, 199.04it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 57 elapsed: 15.542s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 76.80it/s]
eval rmse: 0.0280
eval mae: 0.0128
eval r2: -0.0931
==============================
train: 100%|██████████| 3061/3061 [00:15<00:00, 191.45it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 58 elapsed: 16.135s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 76.80it/s]
eval rmse: 0.0280
eval mae: 0.0133
eval r2: -0.0991
==============================
train: 100%|██████████| 3061/3061 [00:15<00:00, 198.43it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 59 elapsed: 15.586s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 71.94it/s]
eval rmse: 0.0279
eval mae: 0.0133
eval r2: -0.0912
==============================
train: 100%|██████████| 3061/3061 [00:15<00:00, 192.58it/s]
eval_pred:   0%|          | 0/18 [00:00<?, ?it/s]Epoch 60 elapsed: 16.045s
train_loss: 0.0001
eval_pred: 100%|██████████| 18/18 [00:00<00:00, 76.79it/s]
eval rmse: 0.0280
eval mae: 0.0128
eval r2: -0.0927
==============================
*******************************************结束！*******************************************
s