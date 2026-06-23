reset_state("FM")
fm = FM("rating", data_info, embed_size=16, n_epochs=50,
lr=0.001, lr_decay=False, reg=None, batch_size=256,
num_neg=1, use_bn=True, dropout_rate=None, tf_sess_config=None)
fm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
metrics=["rmse", "mae", "r2"])

D:\Develop\Python37\python.exe C:\Users\73556\AppData\Roaming\JetBrains\IntelliJIdea2021.2\plugins\python\helpers\pydev\pydevconsole.py --mode=client --port=58502
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\IdeaSpace\\PythonMusicRecommend', 'D:/IdeaSpace/PythonMusicRecommend'])
PyDev console: starting.
Python 3.7.2 (tags/v3.7.2:9a3ffc0492, Dec 23 2018, 23:09:28) [MSC v.1916 64 bit (AMD64)] on win32
runfile('D:/IdeaSpace/PythonMusicRecommend/LibRecommender-master/test/librec_feat.py', wdir='D:/IdeaSpace/PythonMusicRecommend/LibRecommender-master/test')
2022-03-04 16:27:57.105203: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-03-04 16:27:57.105349: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
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
2022-03-04 16:28:12.184406: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-03-04 16:28:12.186743: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'nvcuda.dll'; dlerror: nvcuda.dll not found
2022-03-04 16:28:12.186873: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2022-03-04 16:28:12.188645: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169] retrieving CUDA diagnostic information for host: ��˹��
2022-03-04 16:28:12.188761: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176] hostname: ��˹��
Training start time: 2022-03-04 16:28:12
total params: 2,550,854 | embedding params: 2,550,801 | network params: 53
train: 100%|██████████| 6416/6416 [01:17<00:00, 82.36it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 1 elapsed: 78.274s
train_loss: 0.0051
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 237.76it/s]
eval rmse: 0.0434
eval mae: 0.0329
eval r2: -0.4802
==============================
train: 100%|██████████| 6416/6416 [01:16<00:00, 83.36it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 2 elapsed: 77.338s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 300.89it/s]
eval rmse: 0.0304
eval mae: 0.0138
eval r2: 0.2734
==============================
train: 100%|██████████| 6416/6416 [01:18<00:00, 81.34it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 3 elapsed: 79.253s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 272.07it/s]
eval rmse: 0.0303
eval mae: 0.0143
eval r2: 0.2802
==============================
train: 100%|██████████| 6416/6416 [01:20<00:00, 79.51it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 4 elapsed: 81.094s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 290.60it/s]
eval rmse: 0.0303
eval mae: 0.0140
eval r2: 0.2770
==============================
train: 100%|██████████| 6416/6416 [01:15<00:00, 84.73it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 5 elapsed: 76.086s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 311.04it/s]
eval rmse: 0.0305
eval mae: 0.0154
eval r2: 0.2690
==============================
train: 100%|██████████| 6416/6416 [01:15<00:00, 84.83it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 6 elapsed: 75.976s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 310.96it/s]
eval rmse: 0.0304
eval mae: 0.0140
eval r2: 0.2746
==============================
train: 100%|██████████| 6416/6416 [01:15<00:00, 85.25it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 7 elapsed: 75.611s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 272.14it/s]
eval rmse: 0.0305
eval mae: 0.0145
eval r2: 0.2668
==============================
train: 100%|██████████| 6416/6416 [01:15<00:00, 84.61it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 8 elapsed: 76.174s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 272.13it/s]
eval rmse: 0.0308
eval mae: 0.0146
eval r2: 0.2556
==============================
train: 100%|██████████| 6416/6416 [01:15<00:00, 85.47it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 9 elapsed: 75.406s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 310.93it/s]
eval rmse: 0.0309
eval mae: 0.0152
eval r2: 0.2477
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 86.09it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 10 elapsed: 74.872s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 310.93it/s]
eval rmse: 0.0311
eval mae: 0.0158
eval r2: 0.2377
==============================
train: 100%|██████████| 6416/6416 [01:13<00:00, 87.30it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 11 elapsed: 73.842s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 310.87it/s]
eval rmse: 0.0310
eval mae: 0.0161
eval r2: 0.2435
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 86.18it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 12 elapsed: 74.780s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 310.89it/s]
eval rmse: 0.0312
eval mae: 0.0160
eval r2: 0.2371
==============================
train: 100%|██████████| 6416/6416 [01:13<00:00, 87.02it/s]
Epoch 13 elapsed: 74.091s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 272.04it/s]
eval rmse: 0.0312
eval mae: 0.0151
eval r2: 0.2336
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 86.23it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 14 elapsed: 74.762s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 272.01it/s]
eval rmse: 0.0312
eval mae: 0.0147
eval r2: 0.2370
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 86.62it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 15 elapsed: 74.414s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 310.96it/s]
eval rmse: 0.0312
eval mae: 0.0154
eval r2: 0.2370
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 86.42it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 16 elapsed: 74.597s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 310.89it/s]
eval rmse: 0.0314
eval mae: 0.0155
eval r2: 0.2240
==============================
train: 100%|██████████| 6416/6416 [35:12<00:00,  3.04it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 17 elapsed: 2112.473s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 362.70it/s]
eval rmse: 0.0312
eval mae: 0.0151
eval r2: 0.2336
==============================
train: 100%|██████████| 6416/6416 [01:15<00:00, 85.18it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 18 elapsed: 75.663s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 332.70it/s]
eval rmse: 0.0313
eval mae: 0.0147
eval r2: 0.2304
==============================
train: 100%|██████████| 6416/6416 [01:15<00:00, 85.50it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 19 elapsed: 75.472s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 362.70it/s]
eval rmse: 0.0316
eval mae: 0.0152
eval r2: 0.2160
==============================
train: 100%|██████████| 6416/6416 [01:16<00:00, 84.12it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 20 elapsed: 76.617s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 310.87it/s]
eval rmse: 0.0317
eval mae: 0.0161
eval r2: 0.2124
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 85.75it/s]
Epoch 21 elapsed: 75.170s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 319.68it/s]
eval rmse: 0.0317
eval mae: 0.0160
eval r2: 0.2082
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 85.94it/s]
Epoch 22 elapsed: 75.019s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 362.85it/s]
eval rmse: 0.0317
eval mae: 0.0159
eval r2: 0.2112
==============================
train: 100%|██████████| 6416/6416 [01:13<00:00, 86.73it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 23 elapsed: 74.321s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 311.17it/s]
eval rmse: 0.0318
eval mae: 0.0154
eval r2: 0.2032
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 86.45it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 24 elapsed: 74.591s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 310.89it/s]
eval rmse: 0.0319
eval mae: 0.0156
eval r2: 0.2001
==============================
train: 100%|██████████| 6416/6416 [01:13<00:00, 87.59it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 25 elapsed: 73.620s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 310.99it/s]
eval rmse: 0.0317
eval mae: 0.0148
eval r2: 0.2094
==============================
train: 100%|██████████| 6416/6416 [01:13<00:00, 87.87it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 26 elapsed: 73.367s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 362.67it/s]
eval rmse: 0.0321
eval mae: 0.0154
eval r2: 0.1891
==============================
train: 100%|██████████| 6416/6416 [01:12<00:00, 88.49it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 27 elapsed: 72.859s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 310.98it/s]
eval rmse: 0.0322
eval mae: 0.0165
eval r2: 0.1847
==============================
train: 100%|██████████| 6416/6416 [01:13<00:00, 87.07it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 28 elapsed: 74.044s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 362.68it/s]
eval rmse: 0.0323
eval mae: 0.0162
eval r2: 0.1817
==============================
train: 100%|██████████| 6416/6416 [01:12<00:00, 87.93it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 29 elapsed: 73.316s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 310.97it/s]
eval rmse: 0.0321
eval mae: 0.0160
eval r2: 0.1923
==============================
train: 100%|██████████| 6416/6416 [01:13<00:00, 87.13it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 30 elapsed: 73.976s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 311.06it/s]
eval rmse: 0.0322
eval mae: 0.0159
eval r2: 0.1866
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 86.49it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 31 elapsed: 74.537s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 362.67it/s]
eval rmse: 0.0324
eval mae: 0.0167
eval r2: 0.1762
==============================
train: 100%|██████████| 6416/6416 [01:13<00:00, 87.18it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 32 elapsed: 73.951s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 310.90it/s]
eval rmse: 0.0321
eval mae: 0.0150
eval r2: 0.1919
==============================
train: 100%|██████████| 6416/6416 [01:12<00:00, 87.92it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 33 elapsed: 73.328s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 362.81it/s]
eval rmse: 0.0323
eval mae: 0.0149
eval r2: 0.1784
==============================
train: 100%|██████████| 6416/6416 [01:13<00:00, 87.55it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 34 elapsed: 73.636s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 362.83it/s]
eval rmse: 0.0322
eval mae: 0.0159
eval r2: 0.1856
==============================
train: 100%|██████████| 6416/6416 [01:12<00:00, 88.51it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 35 elapsed: 72.830s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 362.91it/s]
eval rmse: 0.0322
eval mae: 0.0152
eval r2: 0.1871
==============================
train: 100%|██████████| 6416/6416 [01:12<00:00, 88.18it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 36 elapsed: 73.126s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 310.87it/s]
eval rmse: 0.0322
eval mae: 0.0155
eval r2: 0.1873
==============================
train: 100%|██████████| 6416/6416 [01:11<00:00, 89.18it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 37 elapsed: 72.297s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 362.68it/s]
eval rmse: 0.0326
eval mae: 0.0164
eval r2: 0.1647
==============================
train: 100%|██████████| 6416/6416 [01:12<00:00, 88.21it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 38 elapsed: 73.090s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 311.10it/s]
eval rmse: 0.0323
eval mae: 0.0155
eval r2: 0.1814
==============================
train: 100%|██████████| 6416/6416 [01:11<00:00, 89.55it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 39 elapsed: 71.996s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 361.22it/s]
eval rmse: 0.0321
eval mae: 0.0149
eval r2: 0.1878
==============================
train: 100%|██████████| 6416/6416 [01:12<00:00, 88.53it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 40 elapsed: 72.819s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 362.69it/s]
eval rmse: 0.0324
eval mae: 0.0154
eval r2: 0.1751
==============================
train: 100%|██████████| 6416/6416 [01:12<00:00, 88.20it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 41 elapsed: 73.139s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 362.79it/s]
eval rmse: 0.0327
eval mae: 0.0164
eval r2: 0.1587
==============================
train: 100%|██████████| 6416/6416 [01:12<00:00, 87.98it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 42 elapsed: 73.279s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 357.25it/s]
eval rmse: 0.0324
eval mae: 0.0161
eval r2: 0.1752
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 75.86it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 43 elapsed: 84.941s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 274.19it/s]
eval rmse: 0.0324
eval mae: 0.0150
eval r2: 0.1743
==============================
train: 100%|██████████| 6416/6416 [01:30<00:00, 71.06it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 44 elapsed: 90.707s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 336.63it/s]
eval rmse: 0.0328
eval mae: 0.0172
eval r2: 0.1561
==============================
train: 100%|██████████| 6416/6416 [01:15<00:00, 84.71it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 45 elapsed: 76.128s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 320.75it/s]
eval rmse: 0.0326
eval mae: 0.0160
eval r2: 0.1652
==============================
train: 100%|██████████| 6416/6416 [01:14<00:00, 85.91it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 46 elapsed: 75.114s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 317.75it/s]
eval rmse: 0.0325
eval mae: 0.0160
eval r2: 0.1699
==============================
train: 100%|██████████| 6416/6416 [01:16<00:00, 83.97it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 47 elapsed: 76.780s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 293.61it/s]
eval rmse: 0.0326
eval mae: 0.0159
eval r2: 0.1658
==============================
train: 100%|██████████| 6416/6416 [01:18<00:00, 81.74it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 48 elapsed: 78.870s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 340.01it/s]
eval rmse: 0.0326
eval mae: 0.0154
eval r2: 0.1645
==============================
train: 100%|██████████| 6416/6416 [01:13<00:00, 87.14it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 49 elapsed: 73.998s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 330.11it/s]
eval rmse: 0.0324
eval mae: 0.0163
eval r2: 0.1738
==============================
train: 100%|██████████| 6416/6416 [01:16<00:00, 83.67it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 50 elapsed: 77.052s
train_loss: 0.0007
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 304.17it/s]
eval rmse: 0.0326
eval mae: 0.0157
eval r2: 0.1648
==============================
0.0313