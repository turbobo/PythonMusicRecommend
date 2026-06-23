D:\Develop\Python37\python.exe C:\Users\73556\AppData\Roaming\JetBrains\IntelliJIdea2021.2\plugins\python\helpers\pydev\pydevconsole.py --mode=client --port=61005
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\IdeaSpace\\PythonMusicRecommend', 'D:/IdeaSpace/PythonMusicRecommend'])
PyDev console: starting.
Python 3.7.2 (tags/v3.7.2:9a3ffc0492, Dec 23 2018, 23:09:28) [MSC v.1916 64 bit (AMD64)] on win32
runfile('D:/IdeaSpace/PythonMusicRecommend/LibRecommender-master/test/librec_pure.py', wdir='D:/IdeaSpace/PythonMusicRecommend/LibRecommender-master/test')
2022-03-05 15:31:31.018317: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-03-05 15:31:31.018444: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
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
============================== SVD ==============================
2022-03-05 15:31:47.695383: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-03-05 15:31:47.699378: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'nvcuda.dll'; dlerror: nvcuda.dll not found
2022-03-05 15:31:47.699616: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2022-03-05 15:31:47.703355: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169] retrieving CUDA diagnostic information for host: ��˹��
2022-03-05 15:31:47.703517: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176] hostname: ��˹��
Training start time: 2022-03-05 15:31:47
train: 100%|██████████| 6416/6416 [01:23<00:00, 76.42it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 1 elapsed: 84.238s
train_loss: 0.0012
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 570.53it/s]
eval rmse: 0.0353
eval mae: 0.0200
eval r2: 0.0228
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.74it/s]
Epoch 2 elapsed: 82.906s
train_loss: 0.001
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 641.52it/s]
eval rmse: 0.0358
eval mae: 0.0199
eval r2: -0.0078
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 76.05it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 3 elapsed: 84.679s
train_loss: 0.0009
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 690.62it/s]
eval rmse: 0.0369
eval mae: 0.0213
eval r2: -0.0697
==============================
train: 100%|██████████| 6416/6416 [01:23<00:00, 76.40it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 4 elapsed: 84.277s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 535.34it/s]
eval rmse: 0.0375
eval mae: 0.0213
eval r2: -0.1074
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.85it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 5 elapsed: 82.733s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 680.03it/s]
eval rmse: 0.0381
eval mae: 0.0220
eval r2: -0.1418
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 78.03it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 6 elapsed: 82.518s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 683.25it/s]
eval rmse: 0.0386
eval mae: 0.0221
eval r2: -0.1691
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.51it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 7 elapsed: 83.067s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 684.82it/s]
eval rmse: 0.0389
eval mae: 0.0224
eval r2: -0.1895
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 76.11it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 8 elapsed: 84.601s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 682.70it/s]
eval rmse: 0.0395
eval mae: 0.0225
eval r2: -0.2232
==============================
train: 100%|██████████| 6416/6416 [01:23<00:00, 77.22it/s]
Epoch 9 elapsed: 83.374s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 653.85it/s]
eval rmse: 0.0396
eval mae: 0.0227
eval r2: -0.2312
==============================
train: 100%|██████████| 6416/6416 [01:25<00:00, 75.17it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 10 elapsed: 85.652s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 644.95it/s]
eval rmse: 0.0400
eval mae: 0.0229
eval r2: -0.2563
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 76.36it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 11 elapsed: 84.345s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 572.95it/s]
eval rmse: 0.0403
eval mae: 0.0231
eval r2: -0.2762
==============================
train: 100%|██████████| 6416/6416 [01:23<00:00, 77.02it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 12 elapsed: 83.621s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 629.58it/s]
eval rmse: 0.0404
eval mae: 0.0231
eval r2: -0.2837
==============================
train: 100%|██████████| 6416/6416 [01:21<00:00, 78.43it/s]
Epoch 13 elapsed: 82.102s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 680.00it/s]
eval rmse: 0.0405
eval mae: 0.0232
eval r2: -0.2892
==============================
train: 100%|██████████| 6416/6416 [01:20<00:00, 79.59it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 14 elapsed: 80.897s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 680.01it/s]
eval rmse: 0.0408
eval mae: 0.0234
eval r2: -0.3079
==============================
train: 100%|██████████| 6416/6416 [01:19<00:00, 81.10it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 15 elapsed: 79.401s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 693.90it/s]
eval rmse: 0.0408
eval mae: 0.0234
eval r2: -0.3108
==============================
train: 100%|██████████| 6416/6416 [01:18<00:00, 81.54it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 16 elapsed: 78.974s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 725.06it/s]
eval rmse: 0.0412
eval mae: 0.0236
eval r2: -0.3367
==============================
train: 100%|██████████| 6416/6416 [01:11<00:00, 89.65it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 17 elapsed: 71.846s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 726.11it/s]
eval rmse: 0.0413
eval mae: 0.0236
eval r2: -0.3407
==============================
train: 100%|██████████| 6416/6416 [01:13<00:00, 87.74it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 18 elapsed: 73.423s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 725.58it/s]
eval rmse: 0.0413
eval mae: 0.0236
eval r2: -0.3383
==============================
train: 100%|██████████| 6416/6416 [01:13<00:00, 87.88it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 19 elapsed: 73.280s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 726.29it/s]
eval rmse: 0.0416
eval mae: 0.0238
eval r2: -0.3594
==============================
train: 100%|██████████| 6416/6416 [01:13<00:00, 86.93it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 20 elapsed: 74.087s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 725.87it/s]
eval rmse: 0.0416
eval mae: 0.0238
eval r2: -0.3606
==============================
train: 100%|██████████| 6416/6416 [01:13<00:00, 87.21it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 21 elapsed: 73.848s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 592.95it/s]
eval rmse: 0.0418
eval mae: 0.0239
eval r2: -0.3734
==============================
train: 100%|██████████| 6416/6416 [01:11<00:00, 89.22it/s]
Epoch 22 elapsed: 72.193s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 544.04it/s]
eval rmse: 0.0418
eval mae: 0.0239
eval r2: -0.3752
==============================
train: 100%|██████████| 6416/6416 [01:12<00:00, 88.69it/s]
Epoch 23 elapsed: 72.597s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 725.41it/s]
eval rmse: 0.0419
eval mae: 0.0239
eval r2: -0.3788
==============================
train: 100%|██████████| 6416/6416 [01:12<00:00, 88.23it/s]
Epoch 24 elapsed: 72.993s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 725.92it/s]
eval rmse: 0.0420
eval mae: 0.0240
eval r2: -0.3884
==============================
train: 100%|██████████| 6416/6416 [01:12<00:00, 87.95it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 25 elapsed: 73.216s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 725.81it/s]
eval rmse: 0.0422
eval mae: 0.0241
eval r2: -0.3983
==============================
train: 100%|██████████| 6416/6416 [01:13<00:00, 87.65it/s]
Epoch 26 elapsed: 73.480s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 725.18it/s]
eval rmse: 0.0422
eval mae: 0.0241
eval r2: -0.3967
==============================
train: 100%|██████████| 6416/6416 [01:12<00:00, 88.29it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 27 elapsed: 72.945s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 1088.04it/s]
eval rmse: 0.0421
eval mae: 0.0240
eval r2: -0.3935
==============================
train: 100%|██████████| 6416/6416 [01:12<00:00, 88.41it/s]
Epoch 28 elapsed: 72.844s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 725.65it/s]
eval rmse: 0.0423
eval mae: 0.0241
eval r2: -0.4045
==============================
train: 100%|██████████| 6416/6416 [01:11<00:00, 89.77it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 29 elapsed: 71.723s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 725.88it/s]
eval rmse: 0.0424
eval mae: 0.0242
eval r2: -0.4146
==============================
train: 100%|██████████| 6416/6416 [01:11<00:00, 89.68it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 30 elapsed: 71.814s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 726.13it/s]
eval rmse: 0.0423
eval mae: 0.0241
eval r2: -0.4085
==============================
train: 100%|██████████| 6416/6416 [01:11<00:00, 89.12it/s]
Epoch 31 elapsed: 72.277s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 725.75it/s]
eval rmse: 0.0426
eval mae: 0.0243
eval r2: -0.4289
==============================
train: 100%|██████████| 6416/6416 [01:12<00:00, 89.11it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 32 elapsed: 72.268s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 726.25it/s]
eval rmse: 0.0427
eval mae: 0.0243
eval r2: -0.4314
==============================
train: 100%|██████████| 6416/6416 [01:11<00:00, 89.39it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 33 elapsed: 72.031s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 726.03it/s]
eval rmse: 0.0426
eval mae: 0.0243
eval r2: -0.4285
==============================
train: 100%|██████████| 6416/6416 [01:11<00:00, 89.56it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 34 elapsed: 71.950s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 725.36it/s]
eval rmse: 0.0426
eval mae: 0.0243
eval r2: -0.4293
==============================
train: 100%|██████████| 6416/6416 [01:12<00:00, 88.28it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 35 elapsed: 72.952s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 1087.99it/s]
eval rmse: 0.0429
eval mae: 0.0244
eval r2: -0.4445
==============================
train: 100%|██████████| 6416/6416 [01:12<00:00, 88.36it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 36 elapsed: 72.877s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 726.75it/s]
eval rmse: 0.0428
eval mae: 0.0244
eval r2: -0.4405
==============================
train: 100%|██████████| 6416/6416 [51:50<00:00,  2.06it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 37 elapsed: 3111.198s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 725.36it/s]
eval rmse: 0.0428
eval mae: 0.0243
eval r2: -0.4374
==============================
train: 100%|██████████| 6416/6416 [01:12<00:00, 88.51it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 38 elapsed: 72.769s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 1088.13it/s]
eval rmse: 0.0429
eval mae: 0.0243
eval r2: -0.4451
==============================
train: 100%|██████████| 6416/6416 [01:11<00:00, 89.56it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 39 elapsed: 71.901s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 725.58it/s]
eval rmse: 0.0428
eval mae: 0.0244
eval r2: -0.4418
==============================
train: 100%|██████████| 6416/6416 [01:12<00:00, 88.01it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 40 elapsed: 73.165s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 725.37it/s]
eval rmse: 0.0429
eval mae: 0.0244
eval r2: -0.4458
==============================
*******************************************结束！*******************************************
