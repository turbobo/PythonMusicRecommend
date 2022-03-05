D:\Develop\Python37\python.exe C:\Users\73556\AppData\Roaming\JetBrains\IntelliJIdea2021.2\plugins\python\helpers\pydev\pydevconsole.py --mode=client --port=62428
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\IdeaSpace\\PythonMusicRecommend', 'D:/IdeaSpace/PythonMusicRecommend'])
PyDev console: starting.
Python 3.7.2 (tags/v3.7.2:9a3ffc0492, Dec 23 2018, 23:09:28) [MSC v.1916 64 bit (AMD64)] on win32
runfile('D:/IdeaSpace/PythonMusicRecommend/LibRecommender-master/test/librec_pure.py', wdir='D:/IdeaSpace/PythonMusicRecommend/LibRecommender-master/test')
2022-03-05 17:48:52.156550: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-03-05 17:48:52.156684: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
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
2022-03-05 17:49:06.551352: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-03-05 17:49:06.553366: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'nvcuda.dll'; dlerror: nvcuda.dll not found
2022-03-05 17:49:06.553490: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2022-03-05 17:49:06.555081: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169] retrieving CUDA diagnostic information for host: ��˹��
2022-03-05 17:49:06.555217: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176] hostname: ��˹��
Training start time: 2022-03-05 17:49:06
train: 100%|██████████| 6416/6416 [01:15<00:00, 85.41it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 1 elapsed: 75.404s
train_loss: 0.0012
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 725.38it/s]
eval rmse: 0.0353
eval mae: 0.0200
eval r2: 0.0220
==============================
train: 100%|██████████| 6416/6416 [01:12<00:00, 88.11it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 2 elapsed: 73.071s
train_loss: 0.001
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 725.72it/s]
eval rmse: 0.0359
eval mae: 0.0200
eval r2: -0.0117
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 85.69it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 3 elapsed: 75.151s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 693.88it/s]
eval rmse: 0.0369
eval mae: 0.0213
eval r2: -0.0723
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 85.67it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 4 elapsed: 75.166s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 507.47it/s]
eval rmse: 0.0375
eval mae: 0.0213
eval r2: -0.1046
==============================
train: 100%|██████████| 6416/6416 [01:16<00:00, 83.83it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 5 elapsed: 76.825s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 478.88it/s]
eval rmse: 0.0381
eval mae: 0.0220
eval r2: -0.1416
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 85.62it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 6 elapsed: 75.224s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 1088.27it/s]
eval rmse: 0.0386
eval mae: 0.0220
eval r2: -0.1681
==============================
train: 100%|██████████| 6416/6416 [01:13<00:00, 87.28it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 7 elapsed: 73.766s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 768.03it/s]
eval rmse: 0.0389
eval mae: 0.0224
eval r2: -0.1869
==============================
train: 100%|██████████| 6416/6416 [01:12<00:00, 88.00it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 8 elapsed: 73.199s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 725.15it/s]
eval rmse: 0.0394
eval mae: 0.0226
eval r2: -0.2209
==============================
train: 100%|██████████| 6416/6416 [01:12<00:00, 88.03it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 9 elapsed: 73.156s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 718.30it/s]
eval rmse: 0.0396
eval mae: 0.0228
eval r2: -0.2320
==============================
train: 100%|██████████| 6416/6416 [01:10<00:00, 90.80it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 10 elapsed: 70.928s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 1088.34it/s]
eval rmse: 0.0398
eval mae: 0.0228
eval r2: -0.2474
==============================
train: 100%|██████████| 6416/6416 [01:11<00:00, 89.20it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 11 elapsed: 72.190s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 725.77it/s]
eval rmse: 0.0401
eval mae: 0.0231
eval r2: -0.2658
==============================
train: 100%|██████████| 6416/6416 [01:11<00:00, 89.82it/s]
Epoch 12 elapsed: 71.705s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 1088.12it/s]
eval rmse: 0.0405
eval mae: 0.0232
eval r2: -0.2860
==============================
train: 100%|██████████| 6416/6416 [01:10<00:00, 90.42it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 13 elapsed: 71.240s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 725.45it/s]
eval rmse: 0.0405
eval mae: 0.0231
eval r2: -0.2872
==============================
train: 100%|██████████| 6416/6416 [01:10<00:00, 90.65it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 14 elapsed: 71.058s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 1088.12it/s]
eval rmse: 0.0407
eval mae: 0.0233
eval r2: -0.3027
==============================
train: 100%|██████████| 6416/6416 [01:11<00:00, 89.44it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 15 elapsed: 71.996s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 726.40it/s]
eval rmse: 0.0408
eval mae: 0.0233
eval r2: -0.3097
==============================
train: 100%|██████████| 6416/6416 [01:10<00:00, 90.40it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 16 elapsed: 71.240s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 726.08it/s]
eval rmse: 0.0411
eval mae: 0.0235
eval r2: -0.3278
==============================
train: 100%|██████████| 6416/6416 [01:11<00:00, 89.59it/s]
Epoch 17 elapsed: 71.892s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 726.17it/s]
eval rmse: 0.0411
eval mae: 0.0235
eval r2: -0.3249
==============================
train: 100%|██████████| 6416/6416 [01:11<00:00, 89.35it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 18 elapsed: 72.084s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 724.63it/s]
eval rmse: 0.0413
eval mae: 0.0236
eval r2: -0.3410
==============================
train: 100%|██████████| 6416/6416 [01:12<00:00, 89.05it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 19 elapsed: 72.304s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 726.25it/s]
eval rmse: 0.0414
eval mae: 0.0237
eval r2: -0.3478
==============================
train: 100%|██████████| 6416/6416 [01:11<00:00, 89.32it/s]
Epoch 20 elapsed: 72.098s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 726.18it/s]
eval rmse: 0.0415
eval mae: 0.0237
eval r2: -0.3560
==============================
*******************************************结束！*******************************************
