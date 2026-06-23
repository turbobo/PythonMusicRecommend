D:\Develop\Python37\python.exe C:\Users\73556\AppData\Roaming\JetBrains\IntelliJIdea2021.2\plugins\python\helpers\pydev\pydevconsole.py --mode=client --port=62662
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\IdeaSpace\\PythonMusicRecommend', 'D:/IdeaSpace/PythonMusicRecommend'])
PyDev console: starting.
Python 3.7.2 (tags/v3.7.2:9a3ffc0492, Dec 23 2018, 23:09:28) [MSC v.1916 64 bit (AMD64)] on win32
runfile('D:/IdeaSpace/PythonMusicRecommend/LibRecommender-master/test/librec_feat.py', wdir='D:/IdeaSpace/PythonMusicRecommend/LibRecommender-master/test')
2022-03-06 15:05:49.571334: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-03-06 15:05:49.571487: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
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
============================== Wide_Deep ==============================
2022-03-06 15:06:04.398544: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-03-06 15:06:04.399727: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'nvcuda.dll'; dlerror: nvcuda.dll not found
2022-03-06 15:06:04.399833: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2022-03-06 15:06:04.402255: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169] retrieving CUDA diagnostic information for host: ��˹��
2022-03-06 15:06:04.402438: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176] hostname: ��˹��
Training start time: 2022-03-06 15:06:04
total params: 2,703,382 | embedding params: 2,551,569 | network params: 151,813
train: 100%|██████████| 6416/6416 [01:23<00:00, 77.23it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 1 elapsed: 83.431s
train_loss: 0.0053
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 77.65it/s]
eval rmse: 0.0307
eval mae: 0.0144
eval r2: 0.2570
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 76.18it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 2 elapsed: 84.582s
train_loss: 0.0009
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 78.03it/s]
eval rmse: 0.0304
eval mae: 0.0141
eval r2: 0.2757
==============================
train: 100%|██████████| 6416/6416 [01:23<00:00, 76.90it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 3 elapsed: 83.785s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 77.72it/s]
eval rmse: 0.0303
eval mae: 0.0144
eval r2: 0.2775
==============================
train: 100%|██████████| 6416/6416 [01:23<00:00, 76.69it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 4 elapsed: 84.022s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 80.60it/s]
eval rmse: 0.0304
eval mae: 0.0141
eval r2: 0.2752
==============================
train: 100%|██████████| 6416/6416 [01:23<00:00, 76.83it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 5 elapsed: 83.877s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 82.54it/s]
eval rmse: 0.0305
eval mae: 0.0140
eval r2: 0.2691
==============================
train: 100%|██████████| 6416/6416 [01:25<00:00, 75.23it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 6 elapsed: 85.640s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 77.13it/s]
eval rmse: 0.0304
eval mae: 0.0151
eval r2: 0.2726
==============================
train: 100%|██████████| 6416/6416 [01:25<00:00, 74.75it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 7 elapsed: 86.184s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 76.23it/s]
eval rmse: 0.0312
eval mae: 0.0180
eval r2: 0.2358
==============================
train: 100%|██████████| 6416/6416 [01:27<00:00, 73.44it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 8 elapsed: 87.740s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 79.25it/s]
eval rmse: 0.0307
eval mae: 0.0151
eval r2: 0.2588
==============================
train: 100%|██████████| 6416/6416 [01:30<00:00, 70.77it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 9 elapsed: 91.030s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 73.91it/s]
eval rmse: 0.0308
eval mae: 0.0150
eval r2: 0.2522
==============================
train: 100%|██████████| 6416/6416 [01:26<00:00, 74.15it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 10 elapsed: 87.010s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.69it/s]
eval rmse: 0.0322
eval mae: 0.0196
eval r2: 0.1876
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 78.13it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 11 elapsed: 82.484s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 85.19it/s]
eval rmse: 0.0307
eval mae: 0.0157
eval r2: 0.2584
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.96it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 12 elapsed: 82.650s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.69it/s]
eval rmse: 0.0320
eval mae: 0.0175
eval r2: 0.1967
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 78.05it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 13 elapsed: 82.549s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.69it/s]
eval rmse: 0.0319
eval mae: 0.0184
eval r2: 0.2000
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.76it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 14 elapsed: 82.868s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 80.60it/s]
eval rmse: 0.0309
eval mae: 0.0160
eval r2: 0.2490
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 76.33it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 15 elapsed: 84.395s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 80.45it/s]
eval rmse: 0.0308
eval mae: 0.0143
eval r2: 0.2539
==============================
train: 100%|██████████| 6416/6416 [01:23<00:00, 77.11it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 16 elapsed: 83.556s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.72it/s]
eval rmse: 0.0311
eval mae: 0.0163
eval r2: 0.2387
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.92it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 17 elapsed: 82.682s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.70it/s]
eval rmse: 0.0314
eval mae: 0.0150
eval r2: 0.2251
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.77it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 18 elapsed: 82.850s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.37it/s]
eval rmse: 0.0312
eval mae: 0.0150
eval r2: 0.2367
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.60it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 19 elapsed: 83.039s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.70it/s]
eval rmse: 0.0311
eval mae: 0.0159
eval r2: 0.2401
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.74it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 20 elapsed: 82.881s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 82.43it/s]
eval rmse: 0.0313
eval mae: 0.0162
eval r2: 0.2301
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.91it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 21 elapsed: 82.706s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.41it/s]
eval rmse: 0.0317
eval mae: 0.0152
eval r2: 0.2112
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.96it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 22 elapsed: 82.643s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.69it/s]
eval rmse: 0.0319
eval mae: 0.0169
eval r2: 0.2005
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.65it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 23 elapsed: 82.976s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.70it/s]
eval rmse: 0.0310
eval mae: 0.0152
eval r2: 0.2436
==============================
train: 100%|██████████| 6416/6416 [01:23<00:00, 77.04it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 24 elapsed: 83.628s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.19it/s]
eval rmse: 0.0313
eval mae: 0.0151
eval r2: 0.2312
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.81it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 25 elapsed: 82.800s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.20it/s]
eval rmse: 0.0312
eval mae: 0.0156
eval r2: 0.2355
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.58it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 26 elapsed: 83.049s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.70it/s]
eval rmse: 0.0314
eval mae: 0.0158
eval r2: 0.2250
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.91it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 27 elapsed: 82.701s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.18it/s]
eval rmse: 0.0323
eval mae: 0.0164
eval r2: 0.1820
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.59it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 28 elapsed: 83.045s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.71it/s]
eval rmse: 0.0324
eval mae: 0.0165
eval r2: 0.1760
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.90it/s]
Epoch 29 elapsed: 82.696s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.70it/s]
eval rmse: 0.0315
eval mae: 0.0163
eval r2: 0.2219
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.78it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 30 elapsed: 82.823s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.12it/s]
eval rmse: 0.0321
eval mae: 0.0154
eval r2: 0.1923
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 78.07it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 31 elapsed: 82.531s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 82.66it/s]
eval rmse: 0.0316
eval mae: 0.0145
eval r2: 0.2163
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.83it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 32 elapsed: 82.774s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 82.68it/s]
eval rmse: 0.0312
eval mae: 0.0155
eval r2: 0.2348
==============================
train: 100%|██████████| 6416/6416 [01:23<00:00, 76.51it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 33 elapsed: 84.195s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 82.32it/s]
eval rmse: 0.0315
eval mae: 0.0159
eval r2: 0.2218
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.94it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 34 elapsed: 82.679s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 82.08it/s]
eval rmse: 0.0319
eval mae: 0.0152
eval r2: 0.1984
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.79it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 35 elapsed: 82.850s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.69it/s]
eval rmse: 0.0323
eval mae: 0.0174
eval r2: 0.1821
==============================
train: 100%|██████████| 6416/6416 [01:21<00:00, 78.51it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 36 elapsed: 82.076s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 82.99it/s]
eval rmse: 0.0322
eval mae: 0.0161
eval r2: 0.1835
==============================
train: 100%|██████████| 6416/6416 [01:21<00:00, 78.52it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 37 elapsed: 82.067s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.70it/s]
eval rmse: 0.0325
eval mae: 0.0177
eval r2: 0.1700
==============================
train: 100%|██████████| 6416/6416 [01:21<00:00, 78.32it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 38 elapsed: 82.263s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.09it/s]
eval rmse: 0.0316
eval mae: 0.0167
eval r2: 0.2130
==============================
train: 100%|██████████| 6416/6416 [01:21<00:00, 78.25it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 39 elapsed: 82.344s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 85.68it/s]
eval rmse: 0.0316
eval mae: 0.0147
eval r2: 0.2156
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 78.09it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 40 elapsed: 82.519s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.45it/s]
eval rmse: 0.0315
eval mae: 0.0157
eval r2: 0.2188
==============================
train: 100%|██████████| 6416/6416 [01:21<00:00, 78.26it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 41 elapsed: 82.340s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.71it/s]
eval rmse: 0.0331
eval mae: 0.0174
eval r2: 0.1396
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 78.13it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 42 elapsed: 82.489s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.69it/s]
eval rmse: 0.0314
eval mae: 0.0157
eval r2: 0.2272
==============================
train: 100%|██████████| 6416/6416 [01:21<00:00, 78.26it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 43 elapsed: 82.350s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.71it/s]
eval rmse: 0.0324
eval mae: 0.0170
eval r2: 0.1769
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 78.11it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 44 elapsed: 82.487s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 80.59it/s]
eval rmse: 0.0353
eval mae: 0.0184
eval r2: 0.0188
==============================
train: 100%|██████████| 6416/6416 [01:21<00:00, 78.27it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 45 elapsed: 82.330s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 80.60it/s]
eval rmse: 0.0318
eval mae: 0.0153
eval r2: 0.2053
==============================
train: 100%|██████████| 6416/6416 [01:21<00:00, 78.27it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 46 elapsed: 82.339s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 80.60it/s]
eval rmse: 0.0317
eval mae: 0.0153
eval r2: 0.2085
==============================
train: 100%|██████████| 6416/6416 [01:31<00:00, 70.31it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 47 elapsed: 91.602s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 72.72it/s]
eval rmse: 0.0319
eval mae: 0.0148
eval r2: 0.2010
==============================
train: 100%|██████████| 6416/6416 [01:29<00:00, 71.57it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 48 elapsed: 90.096s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 67.25it/s]
eval rmse: 0.0316
eval mae: 0.0157
eval r2: 0.2170
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 76.29it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 49 elapsed: 84.566s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 72.29it/s]
eval rmse: 0.0323
eval mae: 0.0167
eval r2: 0.1816
==============================
train: 100%|██████████| 6416/6416 [01:23<00:00, 76.76it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 50 elapsed: 83.928s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 80.59it/s]
eval rmse: 0.0316
eval mae: 0.0158
eval r2: 0.2129
==============================
train: 100%|██████████| 6416/6416 [01:21<00:00, 78.47it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 51 elapsed: 82.114s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.70it/s]
eval rmse: 0.0333
eval mae: 0.0192
eval r2: 0.1269
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.70it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 52 elapsed: 82.921s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.69it/s]
eval rmse: 0.0324
eval mae: 0.0161
eval r2: 0.1767
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.95it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 53 elapsed: 82.672s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 80.59it/s]
eval rmse: 0.0317
eval mae: 0.0161
eval r2: 0.2116
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 78.04it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 54 elapsed: 82.572s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.24it/s]
eval rmse: 0.0317
eval mae: 0.0148
eval r2: 0.2084
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.97it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 55 elapsed: 82.647s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 80.34it/s]
eval rmse: 0.0333
eval mae: 0.0181
eval r2: 0.1292
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.89it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 56 elapsed: 82.735s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.41it/s]
eval rmse: 0.0319
eval mae: 0.0152
eval r2: 0.1993
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.97it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 57 elapsed: 82.648s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.69it/s]
eval rmse: 0.0321
eval mae: 0.0160
eval r2: 0.1897
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.85it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 58 elapsed: 82.774s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 80.59it/s]
eval rmse: 0.0317
eval mae: 0.0148
eval r2: 0.2095
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 78.11it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 59 elapsed: 82.476s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.70it/s]
eval rmse: 0.0317
eval mae: 0.0149
eval r2: 0.2112
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.68it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 60 elapsed: 82.945s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.69it/s]
eval rmse: 0.0318
eval mae: 0.0145
eval r2: 0.2071
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 78.10it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 61 elapsed: 82.496s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.10it/s]
eval rmse: 0.0321
eval mae: 0.0164
eval r2: 0.1918
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 78.00it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 62 elapsed: 82.606s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 82.21it/s]
eval rmse: 0.0317
eval mae: 0.0154
eval r2: 0.2087
==============================
train: 100%|██████████| 6416/6416 [01:21<00:00, 79.00it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 63 elapsed: 81.565s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.70it/s]
eval rmse: 0.0339
eval mae: 0.0169
eval r2: 0.0982
==============================
train: 100%|██████████| 6416/6416 [01:21<00:00, 78.74it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 64 elapsed: 81.830s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.70it/s]
eval rmse: 0.0373
eval mae: 0.0203
eval r2: -0.0943
==============================
train: 100%|██████████| 6416/6416 [01:21<00:00, 78.76it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 65 elapsed: 81.812s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 84.14it/s]
eval rmse: 0.0324
eval mae: 0.0153
eval r2: 0.1731
==============================
train: 100%|██████████| 6416/6416 [01:21<00:00, 78.54it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 66 elapsed: 82.041s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 80.59it/s]
eval rmse: 0.0324
eval mae: 0.0167
eval r2: 0.1726
==============================
train: 100%|██████████| 6416/6416 [01:21<00:00, 78.92it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 67 elapsed: 81.627s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.69it/s]
eval rmse: 0.0322
eval mae: 0.0151
eval r2: 0.1837
==============================
train: 100%|██████████| 6416/6416 [04:20<00:00, 24.64it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 68 elapsed: 260.746s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 82.52it/s]
eval rmse: 0.0346
eval mae: 0.0189
eval r2: 0.0584
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 76.08it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 69 elapsed: 84.703s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 81.62it/s]
eval rmse: 0.0318
eval mae: 0.0152
eval r2: 0.2033
==============================
train: 100%|██████████| 6416/6416 [01:24<00:00, 75.93it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 70 elapsed: 84.878s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 81.73it/s]
eval rmse: 0.0338
eval mae: 0.0182
eval r2: 0.1033
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 78.08it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 71 elapsed: 82.539s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 82.73it/s]
eval rmse: 0.0330
eval mae: 0.0174
eval r2: 0.1419
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.82it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 72 elapsed: 82.803s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 83.03it/s]
eval rmse: 0.0324
eval mae: 0.0170
eval r2: 0.1744
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.84it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 73 elapsed: 82.786s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 82.52it/s]
eval rmse: 0.0327
eval mae: 0.0162
eval r2: 0.1584
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.67it/s]
Epoch 74 elapsed: 82.966s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 82.72it/s]
eval rmse: 0.0324
eval mae: 0.0163
eval r2: 0.1732
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.83it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 75 elapsed: 82.798s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 82.44it/s]
eval rmse: 0.0321
eval mae: 0.0152
eval r2: 0.1878
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.45it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 76 elapsed: 83.197s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 82.44it/s]
eval rmse: 0.0321
eval mae: 0.0150
eval r2: 0.1903
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.71it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 77 elapsed: 82.917s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 82.80it/s]
eval rmse: 0.0319
eval mae: 0.0158
eval r2: 0.1980
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.37it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 78 elapsed: 83.280s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 81.90it/s]
eval rmse: 0.0348
eval mae: 0.0197
eval r2: 0.0504
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.60it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 79 elapsed: 83.039s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 82.43it/s]
eval rmse: 0.0320
eval mae: 0.0142
eval r2: 0.1930
==============================
train: 100%|██████████| 6416/6416 [01:22<00:00, 77.64it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]Epoch 80 elapsed: 82.997s
train_loss: 0.0008
eval_pred: 100%|██████████| 34/34 [00:00<00:00, 82.45it/s]
eval rmse: 0.0327
eval mae: 0.0155
eval r2: 0.1601
==============================
*******************************************结束！*******************************************
