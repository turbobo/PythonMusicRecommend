============================== user_cf ==============================
Training start time: 2022-03-05 15:03:57
Final block size and num: (4640, 9)
sim_matrix elapsed: 12.415s
sim_matrix, shape: (41756, 41756), num_elements: 224573284, sparsity: 12.8801 %
top_k: 100%|██████████| 41756/41756 [00:59<00:00, 707.41it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]No common interaction or similar neighbor for user 0 and item 76468, proceed with default prediction
No common interaction or similar neighbor for user 1 and item 36043, proceed with default prediction
No common interaction or similar neighbor for user 2 and item 6100, proceed with default prediction
No common interaction or similar neighbor for user 4 and item 95023, proceed with default prediction
No common interaction or similar neighbor for user 5 and item 96998, proceed with default prediction
No common interaction or similar neighbor for user 8 and item 83107, proceed with default prediction
eval_pred: 100%|██████████| 34/34 [00:29<00:00,  1.15it/s]
	 eval rmse: 0.0387
	 eval mae: 0.0201
	 eval r2: -0.1786
==============================
 ============================== item_cf ==============================
Training start time: 2022-03-05 15:05:38
Final block size and num: (1835, 59)
sim_matrix elapsed: 29.368s
sim_matrix, shape: (108210, 108210), num_elements: 112817004, sparsity: 6.4705 %
top_k: 100%|██████████| 108210/108210 [00:26<00:00, 4159.85it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]No common interaction or similar neighbor for user 0 and item 76468, proceed with default prediction
No common interaction or similar neighbor for user 1 and item 36043, proceed with default prediction
No common interaction or similar neighbor for user 2 and item 6100, proceed with default prediction
No common interaction or similar neighbor for user 4 and item 95023, proceed with default prediction
No common interaction or similar neighbor for user 5 and item 96998, proceed with default prediction
No common interaction or similar neighbor for user 8 and item 83107, proceed with default prediction
eval_pred: 100%|██████████| 34/34 [00:21<00:00,  1.62it/s]
	 eval rmse: 0.0360
	 eval mae: 0.0150
	 eval r2: -0.0197
==============================
结束！


D:\Develop\Python37\python.exe C:\Users\73556\AppData\Roaming\JetBrains\IntelliJIdea2021.2\plugins\python\helpers\pydev\pydevconsole.py --mode=client --port=55251
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\IdeaSpace\\PythonMusicRecommend', 'D:/IdeaSpace/PythonMusicRecommend'])
PyDev console: starting.
Python 3.7.2 (tags/v3.7.2:9a3ffc0492, Dec 23 2018, 23:09:28) [MSC v.1916 64 bit (AMD64)] on win32
runfile('D:/IdeaSpace/PythonMusicRecommend/LibRecommender-master/test/librec_pure.py', wdir='D:/IdeaSpace/PythonMusicRecommend/LibRecommender-master/test')
2022-03-05 15:03:39.082031: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-03-05 15:03:39.082322: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
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
 ============================== user_cf ==============================
Training start time: 2022-03-05 15:03:57
Final block size and num: (4640, 9)
sim_matrix elapsed: 12.415s
sim_matrix, shape: (41756, 41756), num_elements: 224573284, sparsity: 12.8801 %
top_k: 100%|██████████| 41756/41756 [00:59<00:00, 707.41it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]No common interaction or similar neighbor for user 0 and item 76468, proceed with default prediction
No common interaction or similar neighbor for user 1 and item 36043, proceed with default prediction
No common interaction or similar neighbor for user 2 and item 6100, proceed with default prediction
No common interaction or similar neighbor for user 4 and item 95023, proceed with default prediction
No common interaction or similar neighbor for user 5 and item 96998, proceed with default prediction
No common interaction or similar neighbor for user 8 and item 83107, proceed with default prediction
eval_pred: 100%|██████████| 34/34 [00:29<00:00,  1.15it/s]
	 eval rmse: 0.0387
	 eval mae: 0.0201
	 eval r2: -0.1786
==============================
 ============================== item_cf ==============================
Training start time: 2022-03-05 15:05:38
Final block size and num: (1835, 59)
sim_matrix elapsed: 29.368s
sim_matrix, shape: (108210, 108210), num_elements: 112817004, sparsity: 6.4705 %
top_k: 100%|██████████| 108210/108210 [00:26<00:00, 4159.85it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]No common interaction or similar neighbor for user 0 and item 76468, proceed with default prediction
No common interaction or similar neighbor for user 1 and item 36043, proceed with default prediction
No common interaction or similar neighbor for user 2 and item 6100, proceed with default prediction
No common interaction or similar neighbor for user 4 and item 95023, proceed with default prediction
No common interaction or similar neighbor for user 5 and item 96998, proceed with default prediction
No common interaction or similar neighbor for user 8 and item 83107, proceed with default prediction
eval_pred: 100%|██████████| 34/34 [00:21<00:00,  1.62it/s]
	 eval rmse: 0.0360
	 eval mae: 0.0150
	 eval r2: -0.0197
==============================
结束！



D:\Develop\Python37\python.exe C:\Users\73556\AppData\Roaming\JetBrains\IntelliJIdea2021.2\plugins\python\helpers\pydev\pydevconsole.py --mode=client --port=60744
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\IdeaSpace\\PythonMusicRecommend', 'D:/IdeaSpace/PythonMusicRecommend'])
PyDev console: starting.
Python 3.7.2 (tags/v3.7.2:9a3ffc0492, Dec 23 2018, 23:09:28) [MSC v.1916 64 bit (AMD64)] on win32
runfile('D:/IdeaSpace/PythonMusicRecommend/LibRecommender-master/test/librec_pure.py', wdir='D:/IdeaSpace/PythonMusicRecommend/LibRecommender-master/test')
2022-03-05 15:25:22.296757: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-03-05 15:25:22.296878: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
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
 ============================== user_cf ==============================
Training start time: 2022-03-05 15:25:38
Final block size and num: (4640, 9)
sim_matrix elapsed: 14.945s
sim_matrix, shape: (41756, 41756), num_elements: 224573284, sparsity: 12.8801 %
top_k: 100%|██████████| 41756/41756 [01:00<00:00, 690.01it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]No common interaction or similar neighbor for user 0 and item 76468, proceed with default prediction
No common interaction or similar neighbor for user 1 and item 36043, proceed with default prediction
No common interaction or similar neighbor for user 2 and item 6100, proceed with default prediction
No common interaction or similar neighbor for user 4 and item 95023, proceed with default prediction
No common interaction or similar neighbor for user 5 and item 96998, proceed with default prediction
No common interaction or similar neighbor for user 8 and item 83107, proceed with default prediction
eval_pred: 100%|██████████| 34/34 [00:31<00:00,  1.07it/s]
	 eval rmse: 0.0387
	 eval mae: 0.0201
	 eval r2: -0.1786
==============================
 ============================== item_cf ==============================
Training start time: 2022-03-05 15:27:25
Final block size and num: (1835, 59)
sim_matrix elapsed: 30.641s
sim_matrix, shape: (108210, 108210), num_elements: 112817004, sparsity: 6.4705 %
top_k: 100%|██████████| 108210/108210 [00:24<00:00, 4402.55it/s]
eval_pred:   0%|          | 0/34 [00:00<?, ?it/s]No common interaction or similar neighbor for user 0 and item 76468, proceed with default prediction
No common interaction or similar neighbor for user 1 and item 36043, proceed with default prediction
No common interaction or similar neighbor for user 2 and item 6100, proceed with default prediction
No common interaction or similar neighbor for user 4 and item 95023, proceed with default prediction
No common interaction or similar neighbor for user 5 and item 96998, proceed with default prediction
No common interaction or similar neighbor for user 8 and item 83107, proceed with default prediction
eval_pred: 100%|██████████| 34/34 [00:22<00:00,  1.54it/s]
	 eval rmse: 0.0360
	 eval mae: 0.0150
	 eval r2: -0.0197
==============================
结束！
