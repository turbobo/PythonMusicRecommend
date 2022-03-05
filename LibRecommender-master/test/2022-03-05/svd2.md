D:\Develop\Python37\python.exe C:\Users\73556\AppData\Roaming\JetBrains\IntelliJIdea2021.2\plugins\python\helpers\pydev\pydevconsole.py --mode=client --port=62015
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\IdeaSpace\\PythonMusicRecommend', 'D:/IdeaSpace/PythonMusicRecommend'])
PyDev console: starting.
Python 3.7.2 (tags/v3.7.2:9a3ffc0492, Dec 23 2018, 23:09:28) [MSC v.1916 64 bit (AMD64)] on win32
runfile('D:/IdeaSpace/PythonMusicRecommend/LibRecommender-master/test/librec_pure.py', wdir='D:/IdeaSpace/PythonMusicRecommend/LibRecommender-master/test')
2022-03-05 17:25:16.028139: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-03-05 17:25:16.028262: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
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
2022-03-05 17:25:30.013365: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-03-05 17:25:30.016039: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'nvcuda.dll'; dlerror: nvcuda.dll not found
2022-03-05 17:25:30.016208: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2022-03-05 17:25:30.017815: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169] retrieving CUDA diagnostic information for host: ��˹��
2022-03-05 17:25:30.017947: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176] hostname: ��˹��
Training start time: 2022-03-05 17:25:30
train: 100%|██████████| 6416/6416 [01:16<00:00, 84.33it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 1 elapsed: 76.369s
train_loss: 0.0012
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 725.35it/s]
eval rmse: 0.0353
eval mae: 0.0200
eval r2: 0.0206
==============================
train: 100%|██████████| 6416/6416 [01:15<00:00, 84.59it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 2 elapsed: 76.112s
train_loss: 0.001
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 679.85it/s]
eval rmse: 0.0358
eval mae: 0.0199
eval r2: -0.0084
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 86.11it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 3 elapsed: 74.793s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 680.02it/s]
eval rmse: 0.0369
eval mae: 0.0214
eval r2: -0.0707
==============================
train: 100%|██████████| 6416/6416 [01:15<00:00, 85.17it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 4 elapsed: 75.603s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 726.00it/s]
eval rmse: 0.0375
eval mae: 0.0213
eval r2: -0.1075
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 86.19it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 5 elapsed: 74.725s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 725.75it/s]
eval rmse: 0.0381
eval mae: 0.0220
eval r2: -0.1380
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 85.56it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 6 elapsed: 75.254s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 653.84it/s]
eval rmse: 0.0387
eval mae: 0.0221
eval r2: -0.1769
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 86.25it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 7 elapsed: 74.671s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 726.59it/s]
eval rmse: 0.0388
eval mae: 0.0223
eval r2: -0.1859
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 86.67it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 8 elapsed: 74.307s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 725.40it/s]
eval rmse: 0.0394
eval mae: 0.0225
eval r2: -0.2177
==============================
train: 100%|██████████| 6416/6416 [01:13<00:00, 86.77it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 9 elapsed: 74.202s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 726.13it/s]
eval rmse: 0.0396
eval mae: 0.0228
eval r2: -0.2328
==============================
train: 100%|██████████| 6416/6416 [01:15<00:00, 85.49it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 10 elapsed: 75.336s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 725.94it/s]
eval rmse: 0.0399
eval mae: 0.0229
eval r2: -0.2511
==============================
train: 100%|██████████| 6416/6416 [01:15<00:00, 85.29it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 11 elapsed: 75.485s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 723.40it/s]
eval rmse: 0.0401
eval mae: 0.0230
eval r2: -0.2622
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 86.07it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 12 elapsed: 74.832s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 725.54it/s]
eval rmse: 0.0405
eval mae: 0.0231
eval r2: -0.2868
==============================
train: 100%|██████████| 6416/6416 [01:15<00:00, 85.46it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 13 elapsed: 75.343s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 725.51it/s]
eval rmse: 0.0405
eval mae: 0.0233
eval r2: -0.2917
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 86.48it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 14 elapsed: 74.479s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 725.35it/s]
eval rmse: 0.0408
eval mae: 0.0234
eval r2: -0.3088
==============================
train: 100%|██████████| 6416/6416 [01:15<00:00, 85.52it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 15 elapsed: 75.278s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 674.76it/s]
eval rmse: 0.0409
eval mae: 0.0234
eval r2: -0.3115
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 86.30it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 16 elapsed: 74.628s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 714.01it/s]
eval rmse: 0.0411
eval mae: 0.0236
eval r2: -0.3297
==============================
train: 100%|██████████| 6416/6416 [01:15<00:00, 84.54it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 17 elapsed: 76.157s
train_loss: 0.0006
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 641.50it/s]
eval rmse: 0.0410
eval mae: 0.0235
eval r2: -0.3232
==============================
