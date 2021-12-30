# -*- coding: utf-8 -*-
"""Factorization Machine - With Side Information_200w.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oczKp8uiYHwYM5pnrzSe8x7SPR33u6mT

因子分解机
200w数据
user: userID
song: songID, year
raitng: userID, songID, rating(0.001)
200w不做筛选，然后根据：
1、取不同比例user
2、取不同比例song
3、同时取不同比例user和song
分析每个方法的比例趋势，再比较三者结果

### Load Library
"""

# from google.colab import drive
# drive.mount('/content/drive')

# !sudo add-apt-repository universe multiverse
# !sudo apt update
# # # !apt install fastFM
# !sudo apt upgrade
# !sudo apt install fastFM

# # !lsb_release -a
# !sudo hwe-support-status --verbose

# !pip install als
# !pip install fastFM
# !pip install surprise

# ! sudo apt-get install python-dev libopenblas-dev

# # Clone the repo including submodules (or clone + `git submodule update --init --recursive`)
# ! git clone --recursive https://github.com/ibayer/fastFM.git

# # # Enter the root directory
# ! cd fastFM
# !pwd
# ! cd fastFM
# # # Install Python dependencies (Cython>=0.22, numpy, pandas, scipy, scikit-learn)
# ! pip install -r ./requirements.txt

# # Compile the C extension.
# ! make                      # build with default python version (python)
# ! PYTHON=python3 make       # build with custom python version (python3)

# # Install fastFM
# ! pip install .
# !python3 --version

# !pip install pandas==0.2
# import pandas as pd
#
# from google.colab import drive
# drive.mount('/content/drive')

# 1.1.5
# pd.__version__

#Load library
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import time
from math import sqrt
import random
import matplotlib as matplt
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, normalize
from fastfm2 import als
from sklearn.metrics import mean_absolute_error, mean_squared_error

from surprise import KNNBasic
from surprise import Reader, Dataset, accuracy
from surprise import Dataset
from surprise.model_selection import cross_validate

"""### Load Data"""

#Load data
path = './drive/MyDrive/data/metadata/'

#Ratings
# ratings = pd.read_csv(path+'ratings.dat', sep='::', header=None, engine='python')
# ratings.columns = ['userId','movieId','rating','timestamp']
# ratings = ratings.drop('timestamp', axis=1)
# ratings = pd.read_csv(path+'user_item_rating_2.csv')   #, sep=',', header=None, engine='python')
ratings = pd.read_csv(path+'user_item_rating_all_200w.csv')   #, sep=',', header=None, engine='python')
ratings.columns = ['user','song','rating']
ratings.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)

# data
# data = pd.read_csv(path+'track_200w.csv')
data = pd.read_csv(path+'track_all_200w.csv')
# data = data[['user','song','play_count','year','tags']]
data.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)
# data.head(10)
data.astype({'user': 'int32', 'song': 'int32', 'play_count': 'int32', 'year': 'int32'})

"""### Filter data"""

# # 字典user_playcounts记录每个用户的播放总量
# user_playcounts = {}
# for user, group in data.groupby('user'):
#     user_playcounts[user] = group['play_count'].sum()
# temp_user = [user for user in user_playcounts.keys() if user_playcounts[user] > 100]
# temp_playcounts = [playcounts for user, playcounts in user_playcounts.items() if playcounts > 100]
# # data = data[data.user.isin(temp_user)]
# ratings = ratings[ratings.user.isin(temp_user)]
# print('歌曲播放量大于100的用户数量占总体用户数量的比例为', str(round(len(temp_user)/len(user_playcounts), 4)*100)+'%')
# print('歌曲播放量大于100的用户产生的播放总量占总体播放总量的比例为', str(round(sum(temp_playcounts) / sum(user_playcounts.values())*100, 4))+'%')
# print('歌曲播放量大于100的用户产生的数据占总体数据的比例为', str(round(len(data[data.user.isin(temp_user)])/len(data)*100, 4))+"%")



# song_playcounts字典，记录每首歌的播放量
song_playcounts = {}
for song, group in data.groupby('song'):
    song_playcounts[song] = group['play_count'].sum()
temp_song = [song for song in song_playcounts.keys() if song_playcounts[song] > 50]
temp_playcounts = [playcounts for song, playcounts in song_playcounts.items() if playcounts > 50]
# data = data[data.song.isin(temp_song)]
ratings = ratings[ratings.song.isin(temp_song)]
print('播放量大于50的歌曲数量占总体歌曲数量的比例为', str(round(len(temp_song)/len(song_playcounts), 4)*100)+'%')
print('播放量大于50的歌曲产生的播放总量占总体播放总量的比例为', str(round(sum(temp_playcounts) / sum(song_playcounts.values())*100, 4))+'%')
print('播放量大于50的歌曲产生的数据占总体数据的比例为', str(round(len(data[data.song.isin(temp_song)])/len(data)*100, 4))+"%")


print("data:****************************************")
data.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)


#songs
# movies = pd.read_csv(path+'movies.dat', sep='::', header=None, engine='python')
# movies.columns = ['movieId','Title','Genres']
songs = data[['song','year']]
# songs = songs.rename(columns={'year': 'song_year'})
# 根据songID去重
songs.drop_duplicates(subset=['song'],keep='first',inplace=True)
songs.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)
# 去除year为0
songs = songs[songs.year != 0]
songs.head(10)
songs.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)

# 评分中year为0的歌曲去除
song_with_year = [song for song in songs.song]
ratings = ratings[ratings.song.isin(song_with_year)]
del(song_with_year)
# song_with_year.head(10)
# song_with_year.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)
#Users
# users = pd.read_csv(path+'users.dat', sep='::', header=None, engine='python')
# users.columns = ['userId','Gender','Age','Occupation','Zip-code']
# users = users.drop('Zip-code', axis=1)
# user没有附加信息
# users = data[['user']]

"""### Data Quality"""

#Data quality
print('Duplicated rows in ratings file: ' + str(ratings.duplicated().sum()))

# n_users = ratings.userId.unique().shape[0]
# n_movies = ratings.movieId.unique().shape[0]
n_users = ratings.user.unique().shape[0]
n_songs = ratings.song.unique().shape[0]

print('Number of users: {}'.format(n_users))
print('Number of songs: {}'.format(n_songs))
print('Sparsity: {:4.3f}%'.format(float(ratings.shape[0]) / float(n_users*n_songs) * 100))

# all
# Number of users: 41740
# Number of songs: 144068
# Sparsity: 0.027%

# user>100
# Number of users: 16837
# Number of songs: 130006
# Sparsity: 0.055%

# song>50
# Number of users: 41588
# Number of songs: 16973
# Sparsity: 0.159%

# user>100 and song>50
# Number of users: 16832
# Number of songs: 16973
# Sparsity: 0.280%

"""### Data Preprocessing"""

# movies.Genres = movies.Genres.str.split('|')

def expand_df(df, lst_cols, fill_value=''):
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, lens)
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, lens)
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .append(df.loc[lens==0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]

ratings.shape
print('ratings*************************')
ratings.shape
print('songs*************************')
songs.head(10)
songs.shape

# movies = expand_df(movies, ['Genres'])
# movies = movies.drop('Title', axis=1)

# ratings = pd.merge(ratings, users, on="userId")
# ratings = pd.merge(ratings, users, on="user")

"""# 评分连接歌曲信息"""

# ratings_ffm = ratings.merge(movies, left_on='movieId', right_on='movieId', how='inner')
# 评分和歌曲连接：1648992 条数据
ratings_ffm = ratings.merge(songs, left_on='song', right_on='song', how='inner')
# ratings_ffm.info()
# ratings_ffm.head(10)

n_users = ratings_ffm.user.unique().shape[0]
n_songs = ratings_ffm.song.unique().shape[0]

print('Number of users: {}'.format(n_users))
print('Number of songs: {}'.format(n_songs))
print('Sparsity: {:4.3f}%'.format(float(ratings_ffm.shape[0]) / float(n_users*n_songs) * 100))

# all
# Number of users: 41740
# Number of songs: 144068
# Sparsity: 0.027%


# user>100
# Number of users: 16837
# Number of songs: 130006
# Sparsity: 0.055%

# song>50
# Number of users: 41588
# Number of songs: 16973
# Sparsity: 0.159%

# user-song
# Number of users: 16832
# Number of songs: 16973
# Sparsity: 0.280%

# df_dummy = pd.get_dummies(ratings_ffm['Genres'])

# df_new = pd.concat([ratings_ffm, df_dummy], axis=1)

# df_final = df_new.groupby(["userId", "movieId", "rating", "Gender", "Age", "Occupation"])[df_new.columns.values[7:]].sum().reset_index()
# df_final = ratings_ffm.groupby(["user", "song", "rating", 'year'])[ratings_ffm.columns.values[5:]].sum().reset_index()

# user列重新编号   1604290 条数据
df_final = ratings_ffm.groupby(["user", "song", "rating", 'year'])[ratings_ffm.columns.values[5:]].sum().reset_index()
df_final.info()

# user>100 1587913条数据
# song>50 1077529条数据

# 统计用户、歌曲数
print( df_final.user.unique().shape[0] ) 
print( df_final.song.unique().shape[0] )

# user>100
# user = 41736  song = 130006

# song>50

# 1604290 条数据
df_final.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)
print('df_final*************************')
df_final.head(30)
# 释放内存
# del(ratings_ffm)
# del(ratings)
# del(data)

"""### Functions used in training factorization machine"""

#subset data
def subsetdata(data, by, subset_quantile):
    filter_standard = data.groupby([by]).size().reset_index(name='counts').counts.quantile(subset_quantile)
    subset_data = data.groupby(by).filter(lambda x: len(x) >= filter_standard)
    
    return filter_standard, subset_data

#split train and test data
def split_testtrain(ratings, fraction):
    #Transform data in matrix format
    colnames = ratings.columns.values
    # new_colnames = ['1_user', '2_movie', '0_rating', '3_gender', '4_age', '5_occupation', 
    #                 '6_Action', '7_Adventure', '8_Animation', "9_Children's", '10_Comedy',
    #                 '11_Crime', '12_Documentary', '13_Drama', '14_Fantasy', '15_Film-Noir', 
    #                 '16_horror', '17_Musical', '18_Mystery', '19_Romance', '20_Sci-Fi', 
    #                 '21_Thriller', '22_War', '23_Western']

    new_colnames = ['1_user', '2_movie', '0_rating','4_year']
    ratings = ratings.rename(index=str, columns=dict(zip(colnames, new_colnames)))
    # ratings = ratings.rename(index=str, columns=eval(colnames))
    
    ratings_df = ratings.to_dict(orient="records")
    
    dv = DictVectorizer()
    ratings_mat = dv.fit_transform(ratings_df).toarray()
    
    #Split data
    x_train, x_test, y_train, y_test = train_test_split(ratings_mat[:,1:], ratings_mat[:,:1], test_size=fraction)
    
    return x_train, x_test, y_train.T[0], y_test.T[0]

#One hot encoding
def OneHotEncoding(train,test):
    encoder = OneHotEncoder(handle_unknown='ignore').fit(train)
    train = encoder.transform(train)
    test = encoder.transform(test)
    return train, test

#Gridsearch for the optimal parameter
def param_selection(X, y, n_folds):
    start = time.time()
    grid_param = {  
    'n_iter' : np.arange(0,120,25)[1:],
    'rank' :  np.arange(2,12,4),
    }
    grid_search = GridSearchCV(als.FMRegression(l2_reg_w=0.1,l2_reg_V=0.1), cv=n_folds, param_grid=grid_param, verbose=10)
    grid_search.fit(X, y)
    grid_search.best_params_
    print(time.time()-start)
    return grid_search.best_params_

# from tqdm import tqdm
# from scipy import sparse

def rec_coverage(x_test, y_test, prediction, rec_num):
    ratings = pd.DataFrame()
    ratings['user'] = x_test[:,0]
    ratings['song'] = x_test[:,1]
    ratings['rating'] = y_test
    
    pred = ratings.copy()
    pred['rating'] = prediction
    
    # # rating_table = pd.pivot_table(ratings, index='user', columns = 'movie', values = 'rating')
    # rating_table = pd.pivot_table(ratings, index='user', columns = 'song', values = 'rating')
    # # pred_table = pd.pivot_table(pred, index='user', columns = 'movie', values = 'rating')
    # pred_table = pd.pivot_table(pred, index='user', columns = 'song', values = 'rating')
    
    # ratings数据分块
    chunk_size = 50000
    chunks = [x for x in range(0, ratings.shape[0], chunk_size)]
    for i in range(0, len(chunks) - 1):
        print(chunks[i], chunks[i + 1] - 1)
    # 0 49999
    # 50000 99999
    # 100000 149999
    # 150000 199999
    # 200000 249990
    # .........................
    pivot_df = pd.DataFrame()
    for i in tqdm(range(0, len(chunks) - 1)):
      chunk_df = ratings.iloc[ chunks[i]:chunks[i + 1] - 1]
      interactions = (chunk_df.groupby(['user', 'song'])['rating']
        .sum()
        .unstack()
        .reset_index()
        .fillna(0)
        .set_index('user')
      )
      # print (interactions.shape)
      pivot_df = pivot_df.append(interactions, sort=False) 

    df_new = pd.concat([ratings.iloc[ chunks[i]:chunks[i + 1] - 1 ].pivot(index='user', columns='song', values='rating') for i in range(0, len(chunks) - 1)])
    rating_table = sparse.coo_matrix(df_new.to_numpy())

    # pred数据分块
    chunk_size = 50000
    chunks = [x for x in range(0, pred.shape[0], chunk_size)]
    for i in range(0, len(chunks) - 1):
        print(chunks[i], chunks[i + 1] - 1)
    # 0 49999
    # 50000 99999
    # 100000 149999
    # 150000 199999
    # 200000 249990
    # .........................
    pivot_df = pd.DataFrame()
    for i in tqdm(range(0, len(chunks) - 1)):
      chunk_df = pred.iloc[ chunks[i]:chunks[i + 1] - 1]
      interactions = (chunk_df.groupby(['user', 'song'])['rating']
        .sum()
        .unstack()
        .reset_index()
        .fillna(0)
        .set_index('user')
      )
      # print (interactions.shape)
      pivot_df = pivot_df.append(interactions, sort=False) 

    df_new = pd.concat([pred.iloc[ chunks[i]:chunks[i + 1] - 1 ].pivot(index='user', columns='song', values='rating') for i in range(0, len(chunks) - 1)])
    pred_table = sparse.coo_matrix(df_new.to_numpy())


    # rec_movies = []
    rec_songs = []
    rec = pred_table - rating_table
    for user in rec.index:
            rec_item = pred_table.loc[user,:].sort_values(ascending = False).head(rec_num).index.tolist()
            rec_songs += rec_item
    n_rec = len(set(rec_songs))
    n_songs = pred_table.shape[1]
    coverage = round(float(n_rec)/n_songs,2)
    
    return coverage

def create_plot(x1, x2, x3, y1, y2, y3, kind):
    pal = sns.color_palette("Set2")
    
    matplt.figure.Figure(figsize=(5000,5000))
    plt.plot(x1, y1, c=pal[0], label="Filter-User", linewidth=3)
    plt.plot(x2, y2, c=pal[1], label="Filter-Movie", linewidth=3)
    plt.plot(x3, y3, c=pal[2], label="Filter-Both", linewidth=3)
    plt.legend(loc='best', fontsize=12)
    plt.xticks(fontsize=12);
    plt.yticks(fontsize=12);
    plt.xlabel("Sampled Data Size", fontsize=14);
    plt.ylabel(kind, fontsize=14);
    plt.title(kind, loc='center', fontsize=16);
    plt.show()

"""### Factorization Machine"""

def FieldFactorizationMachine(ratings, subset_by, subset_quantile, op_iter, op_rank):
    #Map value
    # gender_dict = {"F": 0, "M": 1}
    # ratings = ratings.replace({"Gender": gender_dict})
    
    #Initialize output
    final_output = pd.DataFrame()
    result_dict = []
    n_iteration = 1 
    last_RMSE = 100
    threshold = 0

    n_users = ratings.user.unique().shape[0]
    n_songs = ratings.song.unique().shape[0]
    n_size = ratings.shape[0]*ratings.shape[1]
    
    sparsity = round(float(ratings.shape[0]) / float(n_users*n_songs),2)
    
    print("---Spliting Test and Train Data---")
    #split test and train data
    xtrain, xtest, ytrain, ytest = split_testtrain(ratings, 0.2)

    print("---Encoding Data---")
    #encode data
    xtrain_enc, xtest_enc = OneHotEncoding(xtrain, xtest)
    
  
    start = time.time()
    print("---Factorization Machine---")
    #Factorization machine
    # op_iter默认100
    fm = als.FMRegression(n_iter=op_iter, rank=op_rank, l2_reg_w=0.1, l2_reg_V=0.1)
    fm.fit(xtrain_enc, ytrain)
    predictions = fm.predict(xtest_enc)
    spent_time = time.time() - start
    #Evaluation metrics
    rmse = sqrt(mean_squared_error(ytest,predictions))
    mae = mean_absolute_error(ytest,predictions)
    print("rmse:")
    print(rmse)
    print("mae:")
    print(mae)
    
    # for quantile in subset_quantile:
    #     print("---Running iteration " + str(n_iteration) + " ---")
    #     print("---Subsetting Original Data---")
        
    #     #subset original data
    #     if subset_by == "user":
    #         filter_standard, subset_ratings = subsetdata(ratings, "user", quantile)
    #     elif subset_by == "song":
    #         filter_standard, subset_ratings = subsetdata(ratings, "song", quantile)
    #     else:
    #         f1, subset_u = subsetdata(ratings, "user", quantile)
    #         f2, subset_ratings = subsetdata(subset_u, "song", quantile)
    #         filter_standard = "("+str(f1)+","+str(f2)+")"
        
    #     n_users = subset_ratings.user.unique().shape[0]
    #     n_songs = subset_ratings.song.unique().shape[0]
    #     n_size = subset_ratings.shape[0]*subset_ratings.shape[1]
        
    #     sparsity = round(float(subset_ratings.shape[0]) / float(n_users*n_songs),2)
        
    #     print("---Spliting Test and Train Data---")
    #     #split test and train data
    #     xtrain, xtest, ytrain, ytest = split_testtrain(subset_ratings, 0.2)

    #     print("---Encoding Data---")
    #     #encode data
    #     xtrain_enc, xtest_enc = OneHotEncoding(xtrain, xtest)
        
       
    #     start = time.time()
    #     print("---Factorization Machine---")
    #     #Factorization machine
    #     fm = als.FMRegression(n_iter=op_iter, rank=op_rank, l2_reg_w=0.1, l2_reg_V=0.1)
    #     fm.fit(xtrain_enc, ytrain)
    #     predictions = fm.predict(xtest_enc)
    #     spent_time = time.time() - start
    #     #Evaluation metrics
    #     rmse = sqrt(mean_squared_error(ytest,predictions))
    #     mae = mean_absolute_error(ytest,predictions)
    #     # 去掉覆盖率计算
    #     # coverage = rec_coverage(xtest, ytest, predictions, 10)
        
    #     if rmse < last_RMSE:
    #         last_RMSE = rmse
    #         threshold = filter_standard
    #         out = pd.DataFrame()
    #         out['user'] = xtest[:,0]
    #         out['song'] = xtest[:,1]
    #         out['rating'] = ytest
    #         out['prediction'] = predictions
            
    #         final_output = out.copy()
        
    #     # result_dict.append([quantile, filter_standard, n_size, n_users, n_songs, sparsity, op_iter, op_rank, spent_time, mae, rmse, coverage])
    #     result_dict.append([quantile, filter_standard, n_size, n_users, n_songs, sparsity, op_iter, op_rank, spent_time, mae, rmse])
    #     n_iteration += 1
    
    # results = pd.DataFrame(result_dict)
    # # results.columns = ["Quantile", "Threshold", "Size", "Num_Users", "Num_Songs", "Sparsity", "OP_Iter", "OP_Rank", "Running Time", "MAE", "RMSE", "Coverage"]
    # results.columns = ["Quantile", "Threshold", "Size", "Num_Users", "Num_Songs", "Sparsity", "OP_Iter", "OP_Rank", "Running Time", "MAE", "RMSE"]
    
    # final_output.to_csv(path+"FFM_Output_"+subset_by+"_"+str(threshold)+".csv", sep=',', encoding='utf-8', index=False)
    
    # return results
    return rmse

quantile_list = np.arange(0.1,1,0.1)
df_final.info()

"""### Subset method 1 - Subset data from less prolific users to prolific users"""

accuracy_matrix_user = FieldFactorizationMachine(df_final, "user", quantile_list, 100, 8)
#n_iter=100, rangk=2 rmse:0.035048356592741685  mae: 0.014596647510181046
#n_iter=100, rangk=8 rmse:0.037385966034999725  mae: 0.016747147277330194

"""# 新段落"""

accuracy_matrix_user

#size_norm_u = normalize(accuracy_matrix_user['Size'][:,np.newaxis], axis=0).ravel()
size_norm_u = 1-np.array(accuracy_matrix_user['Quantile'])

time_u = np.array(accuracy_matrix_user['Running Time'])
mae_u = np.array(accuracy_matrix_user['MAE'])
rmse_u = np.array(accuracy_matrix_user['RMSE'])
# coverage_u = np.array(accuracy_matrix_user['Coverage'])

"""### Subset method 2 - Subset data from less popular items to popular items"""

# df_final.head()
# del(accuracy_matrix_user)
accuracy_matrix_song = FieldFactorizationMachine(ratings, "song", quantile_list, 25, 2)
# accuracy_matrix_song = FieldFactorizationMachine(ratings, "song", quantile_list, 100, 2)

accuracy_matrix_song

#size_norm_m = normalize(accuracy_matrix_movie['Size'][:,np.newaxis], axis=0).ravel()
size_norm_m = 1-np.array(accuracy_matrix_song['Quantile'])

time_m = np.array(accuracy_matrix_song['Running Time'])
mae_m = np.array(accuracy_matrix_song['MAE'])
rmse_m = np.array(accuracy_matrix_song['RMSE'])
# coverage_m = np.array(accuracy_matrix_song['Coverage'])

"""### Subset method 3 - Subset data in both user and item directions"""

accuracy_matrix_both = FieldFactorizationMachine(ratings, "both", quantile_list, 25, 2)

accuracy_matrix_both

#a = normalize(accuracy_matrix_both['Size'][:,np.newaxis], axis=0).ravel()
size_norm_b = 1-np.array(accuracy_matrix_both['Quantile'])

time_b = np.array(accuracy_matrix_both['Running Time'])
mae_b = np.array(accuracy_matrix_both['MAE'])
rmse_b = np.array(accuracy_matrix_both['RMSE'])
# coverage_b = np.array(accuracy_matrix_both['Coverage'])

"""### Benchmark model - Collaborative Filtering Using k-Nearest Neighbors (kNN)"""

# KNNdata = Dataset.load_builtin('ml-1m')

# algo = KNNBasic()

# cross_validate(algo, KNNdata, measures = ['MAE','RMSE'], cv = 3, verbose = True)
path = './drive/MyDrive/data/metadata/'
user_item_rating = pd.read_csv(path+'user_item_rating_2.csv')
# user_item_rating = pd.read_csv(path+'user_item_rating_all_200w.csv')

# In[]
# itemCF

# 阅读器
reader = Reader(line_format='user item rating', sep=',')
# 载入数据
raw_data = Dataset.load_from_df(user_item_rating, reader=reader)
# cross_validate交叉验证
algo = KNNBasic()
cross_validate(algo, raw_data, measures = ['MAE','MSE'], cv = 3, verbose = True)

"""### Evaluation"""

create_plot(size_norm_u, size_norm_m, size_norm_b, time_u, time_m, time_b, "Running Time")

create_plot(size_norm_u, size_norm_m, size_norm_b, mae_u, mae_m, mae_b, "Mean Average Error")

create_plot(size_norm_u, size_norm_m, size_norm_b, rmse_u, rmse_m, rmse_b, "Root Mean Square Error")

# create_plot(size_norm_u, size_norm_m, size_norm_b, coverage_u, coverage_m, coverage_b, "Coverage")