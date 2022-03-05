D:\Develop\Python37\python.exe C:\Users\73556\AppData\Roaming\JetBrains\IntelliJIdea2021.2\plugins\python\helpers\pydev\pydevconsole.py --mode=client --port=63846
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\IdeaSpace\\PythonMusicRecommend', 'D:/IdeaSpace/PythonMusicRecommend'])
PyDev console: starting.
Python 3.7.2 (tags/v3.7.2:9a3ffc0492, Dec 23 2018, 23:09:28) [MSC v.1916 64 bit (AMD64)] on win32
runfile('D:/IdeaSpace/PythonMusicRecommend/LibRecommender-master/test/librec_feat.py', wdir='D:/IdeaSpace/PythonMusicRecommend/LibRecommender-master/test')
2022-03-05 18:33:31.213680: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-03-05 18:33:31.213832: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
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
============================== FM ==============================
2022-03-05 18:33:46.176313: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-03-05 18:33:46.178389: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'nvcuda.dll'; dlerror: nvcuda.dll not found
2022-03-05 18:33:46.178521: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2022-03-05 18:33:46.180300: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169] retrieving CUDA diagnostic information for host: ��˹��
2022-03-05 18:33:46.180429: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176] hostname: ��˹��
Training start time: 2022-03-05 18:33:46
total params: 2,550,854 | embedding params: 2,550,801 | network params: 53
train: 100%|██████████| 6416/6416 [01:18<00:00, 81.58it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 1 elapsed: 79.027s
train_loss: 0.0051
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 215.52it/s]
eval rmse: 0.0434
eval mae: 0.0329
eval r2: -0.4802
==============================
train: 100%|██████████| 6416/6416 [01:16<00:00, 83.55it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 2 elapsed: 77.178s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 310.89it/s]
eval rmse: 0.0304
eval mae: 0.0138
eval r2: 0.2734
==============================
train: 100%|██████████| 6416/6416 [01:18<00:00, 81.73it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 3 elapsed: 78.868s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 272.01it/s]
eval rmse: 0.0303
eval mae: 0.0143
eval r2: 0.2802
==============================
train: 100%|██████████| 6416/6416 [01:16<00:00, 83.36it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 4 elapsed: 77.365s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 304.81it/s]
eval rmse: 0.0303
eval mae: 0.0140
eval r2: 0.2770
==============================
train: 100%|██████████| 6416/6416 [01:16<00:00, 83.46it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 5 elapsed: 77.224s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 310.93it/s]
eval rmse: 0.0305
eval mae: 0.0154
eval r2: 0.2690
==============================
train: 100%|██████████| 6416/6416 [01:15<00:00, 84.59it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 6 elapsed: 76.186s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 311.03it/s]
eval rmse: 0.0304
eval mae: 0.0140
eval r2: 0.2746
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 85.99it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 7 elapsed: 74.970s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 310.85it/s]
eval rmse: 0.0305
eval mae: 0.0145
eval r2: 0.2668
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 85.67it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 8 elapsed: 75.236s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 310.99it/s]
eval rmse: 0.0308
eval mae: 0.0146
eval r2: 0.2556
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 85.89it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 9 elapsed: 75.039s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 310.88it/s]
eval rmse: 0.0309
eval mae: 0.0152
eval r2: 0.2477
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 85.88it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 10 elapsed: 75.057s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 310.87it/s]
eval rmse: 0.0311
eval mae: 0.0158
eval r2: 0.2377
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 85.75it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 11 elapsed: 75.162s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 301.58it/s]
eval rmse: 0.0310
eval mae: 0.0161
eval r2: 0.2435
==============================
train: 100%|██████████| 6416/6416 [01:15<00:00, 85.40it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 12 elapsed: 75.468s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 310.84it/s]
eval rmse: 0.0312
eval mae: 0.0160
eval r2: 0.2371
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 86.05it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 13 elapsed: 74.914s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 272.13it/s]
eval rmse: 0.0312
eval mae: 0.0151
eval r2: 0.2336
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 85.77it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 14 elapsed: 75.149s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 272.01it/s]
eval rmse: 0.0312
eval mae: 0.0147
eval r2: 0.2370
==============================
train: 100%|██████████| 6416/6416 [01:15<00:00, 85.29it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 15 elapsed: 75.575s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 272.04it/s]
eval rmse: 0.0312
eval mae: 0.0154
eval r2: 0.2370
==============================
train: 100%|██████████| 6416/6416 [01:15<00:00, 85.39it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 16 elapsed: 75.488s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 272.15it/s]
eval rmse: 0.0314
eval mae: 0.0155
eval r2: 0.2240
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 86.53it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 17 elapsed: 74.494s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 272.01it/s]
eval rmse: 0.0312
eval mae: 0.0151
eval r2: 0.2336
==============================
train: 100%|██████████| 6416/6416 [24:10<00:00,  4.42it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 18 elapsed: 1450.457s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 304.17it/s]
eval rmse: 0.0313
eval mae: 0.0147
eval r2: 0.2304
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 86.00it/s]
Epoch 19 elapsed: 74.950s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 310.87it/s]
eval rmse: 0.0316
eval mae: 0.0152
eval r2: 0.2160
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 85.98it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 20 elapsed: 74.968s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 310.87it/s]
eval rmse: 0.0317
eval mae: 0.0161
eval r2: 0.2124
==============================
train: 100%|██████████| 6416/6416 [01:15<00:00, 85.21it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 21 elapsed: 75.639s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 272.03it/s]
eval rmse: 0.0317
eval mae: 0.0160
eval r2: 0.2082
==============================
train: 100%|██████████| 6416/6416 [01:15<00:00, 85.39it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 22 elapsed: 75.479s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 311.03it/s]
eval rmse: 0.0317
eval mae: 0.0159
eval r2: 0.2112
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 86.18it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 23 elapsed: 74.790s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 311.02it/s]
eval rmse: 0.0318
eval mae: 0.0154
eval r2: 0.2032
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 85.79it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 24 elapsed: 75.146s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 272.01it/s]
eval rmse: 0.0319
eval mae: 0.0156
eval r2: 0.2001
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 85.67it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 25 elapsed: 75.227s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 310.94it/s]
eval rmse: 0.0317
eval mae: 0.0148
eval r2: 0.2094
==============================
