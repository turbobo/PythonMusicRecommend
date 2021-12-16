
# coding: utf-8

# # Load Necessary Dependencies

# In[1]:

# 第三方库在本地位置  E:\Python\Lib\site-packages
import pandas as pd
import numpy as np
import time
import sqlite3

data_home = './'


# # Load and Process the Datasets

# ### Get more information about the Millionsong project from https://labrosa.ee.columbia.edu/millionsong/
# 
# #### Refer to Chapter 10: Section 'The Million Song Dataset Taste Profile' for more details

# ## Load Triplets data  [user, song, play_count]

# #### Get the data from http://labrosa.ee.columbia.edu/millionsong/sites/default/files/challenge/train_triplets.txt.zip

# In[2]:

# 这个数据大概包含了大约1019319用户对384547首歌的48373585条播放记录。
# triplet_dataset = pd.read_csv(filepath_or_buffer=data_home+'train_triplets.txt',
#                               nrows=10000,sep='\t', header=None,
#                               names=['user','song','play_count'])
# triplet_dataset_all = pd.read_csv(filepath_or_buffer=data_home+'train_triplets.txt')


# In[3]:


# triplet_dataset.head(n=10)


# ## Get User and total play counts

# In[5]:

# 统计 用户-播放歌曲次数
# 对于这样规模大小的数据集，我们首先要做的是有多少用户(或者歌曲)是我们应该要考虑的。
# 在原始数据集中，有大约100万的用户，但是这里面是不是所有用户我们都需要纳入考虑呢？
# 比如说，如果20%的用户的歌曲播放了占了80%的总体播放量，那么其实我们只需要考虑这20%用户就差不多了。
# 一般来说，我们统计一下播放量的累积求和就可以知道多少用户占了80%的总体播放量。
# 不过鉴于数据量如此之大，pandas提供的累积求和功能会出问题。
# 所以我们必须自己一行行地读取这个文件，一部分一部分地来完成这项工作：

# output_dict = {}
# with open(data_home+'train_triplets.txt') as f:
#     for line_number, line in enumerate(f):
#         user = line.split('\t')[0]
#         play_count = int(line.split('\t')[2])
#         if user in output_dict:
#             play_count +=output_dict[user]
#             output_dict.update({user:play_count})
#         output_dict.update({user:play_count})
# output_list = [{'user':k,'play_count':v} for k,v in output_dict.items()]
# play_count_df = pd.DataFrame(output_list)
# play_count_df = play_count_df.sort_values(by = 'play_count', ascending = False)


# In[ ]:


# play_count_df.to_csv(path_or_buf='user_playcount_df.csv', index = False)


# ## Get Song and total play counts

# In[7]:

# 已写入： 歌曲-播放歌曲次数
# output_dict = {}
# with open(data_home+'train_triplets.txt') as f:
#     for line_number, line in enumerate(f):
#         song = line.split('\t')[1]
#         play_count = int(line.split('\t')[2])
#         if song in output_dict:
#             play_count +=output_dict[song]
#             output_dict.update({song:play_count})
#         output_dict.update({song:play_count})
# output_list = [{'song':k,'play_count':v} for k,v in output_dict.items()]
# song_count_df = pd.DataFrame(output_list)
# song_count_df = song_count_df.sort_values(by = 'play_count', ascending = False)


# In[ ]:


# song_count_df.to_csv(path_or_buf='song_playcount_df.csv', index = False)


# ## View top users and songs

# In[14]:


play_count_df = pd.read_csv(filepath_or_buffer='data/metadata/user_playcount_df.csv')
play_count_df.head(n =10)


# In[15]:


song_count_df = pd.read_csv(filepath_or_buffer='data/metadata/song_playcount_df.csv')
song_count_df.head(10)


# ## Subsetting the data

# In[15]:

# 有了这两份数据，我们首要的就是要找到前多少用户占了40%的总体播放量。
# 这个"40%"是我们随机选的一个值，大家在实际的工作中可以自己选择这个数值，重点是控制数据集的大小。
# 当然，如果有高效的Presto(支持HiveQL，但纯内存计算)集群的话，在整体数据集上统计这样的数据也会很快。
# 就我们这个数据集，大约前100,000用户的播放量占据了总体的40%。
# 播放量前10w的用户的总播放量  占  总播放量(138680243次播放)的  40.88%  ==> 10%的用户占据了40%的播放量。
total_play_count = sum(song_count_df.play_count)
# 56693493/138680243 = 40.88%
(float(play_count_df.head(n=100000).play_count.sum())/total_play_count)*100
# 播放量前40w的用户的总播放量  占  总播放量(138680243次播放)的  79.9%  ==> 40%的用户占据了80%的播放量。
# (float(play_count_df.head(n=400000).play_count.sum())/total_play_count)*100 = 79.90847982578167
# play_count_subset = play_count_df.head(n=100000)
# 前3w用户占19.19%
# play_count_subset = play_count_df.head(n=30000)

# 随机获取 30000用户
# DataFrame.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)[source]
# n：要抽取的行数	 frac：抽取行的比例
# replace：True:取行数据后，可以重复放回后再取，False:取行数据后不放回，下次取其它行数据
# weights：字符索引或概率数组  axis=0:为行字符索引或概率数组  axis=1:为列字符索引或概率数组
# random_state：random_state=None,取得数据不重复 random_state=1,可以取得重复数据
# axis=0:抽取行 axis=1:抽取列
play_count_subset = play_count_df.sample(n=8000, frac=None, replace=True, weights=None, random_state=1, axis=0)


# In[17]:

# 同样的，我们发现大约30,000首歌占据了总体80%的播放量。这个信息就很有价值：10%的歌曲占据了80%的播放量。
# 那么，通过这样一些条件，我们就可以从原始的数据集中抽取出最具代表性的数据出来，从而使得需要处理的数据量在一个可控的范围内。
# 108715816/138680243 = 0.7839
(float(song_count_df.head(n=30000).play_count.sum())/total_play_count)*100


# In[18]:

# 歌曲播放量前3w的歌曲
# song_count_subset = song_count_df.head(n=30000)
# 前6000首歌占50.19%
# song_count_subset = song_count_df.head(n=6000)

# 随机获取 6000歌曲
song_count_subset = song_count_df.sample(n=4000, frac=None, replace=True, weights=None, random_state=1, axis=0)



# In[19]:


user_subset = list(play_count_subset.user)
song_subset = list(song_count_subset.song)


# In[20]:

# 已写入：前10w用户、3w歌曲子集
# triplet_dataset = pd.read_csv(filepath_or_buffer=data_home+'train_triplets.txt',sep='\t',
# #                               header=None, names=['user','song','play_count'])
# # triplet_dataset_sub = triplet_dataset[triplet_dataset.user.isin(user_subset) ]
# # del(triplet_dataset)
# # triplet_dataset_sub_song = triplet_dataset_sub[triplet_dataset_sub.song.isin(song_subset)]
# # del(triplet_dataset_sub)

# 测试数据已写入：前3w用户、6000歌曲子集
triplet_dataset = pd.read_csv(filepath_or_buffer=data_home+'train_triplets.txt',sep='\t',
                              header=None, names=['user','song','play_count'])
triplet_dataset_sub = triplet_dataset[triplet_dataset.user.isin(user_subset)]
# 用完后释放内存
# del(triplet_dataset)

# 只获取上面用户播放过的歌曲
# triplet_dataset_sub_song = triplet_dataset_sub[triplet_dataset_sub.song.isin(song_subset)]

# 直接获取元数据集中 存在于song_subset的歌曲
triplet_dataset_sub_song = triplet_dataset[triplet_dataset.song.isin(song_subset)]
del(triplet_dataset)
del(triplet_dataset_sub)


# In[ ]:


# triplet_dataset_sub_song.to_csv(path_or_buf=data_home+'triplet_dataset_sub_song.csv', index=False)
# 2409809
triplet_dataset_sub_song.to_csv(path_or_buf=data_home+'triplet_dataset_sub_song_test.csv', index=False)


# In[25]:

# 已写入：直接从文件读出来
# triplet_dataset_sub_song = pd.read_csv(filepath_or_buffer='triplet_dataset_sub_song.csv')

triplet_dataset_sub_song = pd.read_csv(filepath_or_buffer='data/metadata/triplet_dataset_sub_song_test.csv')
triplet_dataset_sub_song.shape


# In[29]:


triplet_dataset_sub_song.head(n=10)


# ## Adding songs metadata from million songs dataset

# #### Get the data from http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/track_metadata.db

# In[45]:


conn = sqlite3.connect(data_home+'track_metadata.db')
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
cur.fetchall()


# In[49]:

# track_metadata_df  100w行 14列
track_metadata_df = pd.read_sql(con=conn, sql='select * from songs')
# track_metadata_df_sub  30447行 14列
track_metadata_df_sub = track_metadata_df[track_metadata_df.song_id.isin(song_subset)]
track_metadata_df.head()


# In[47]:





# In[50]:

# 已写入
track_metadata_df_sub.to_csv(path_or_buf=data_home+'track_metadata_df_sub_test.csv', index=False)


# In[51]:


track_metadata_df_sub.shape


# ## Load up the saved data subsets

# In[2]:

# 原代码
# triplet_dataset_sub_song = pd.read_csv(filepath_or_buffer=data_home+'train_triplets_sub_song.csv')
# track_metadata_df_sub = pd.read_csv(filepath_or_buffer=data_home+'track_metadata_df_sub.csv')

# jupyter代码
# triplet_dataset_sub_song = pd.read_csv(filepath_or_buffer=data_home+'triplet_dataset_sub_song.csv',encoding = "ISO-8859-1")
triplet_dataset_sub_song = pd.read_csv(filepath_or_buffer=data_home+'triplet_dataset_sub_song_test.csv',encoding = "ISO-8859-1")
track_metadata_df_sub = pd.read_csv(filepath_or_buffer=data_home+'track_metadata_df_sub_test.csv',encoding = "ISO-8859-1")


# ## Clean up datasets

# In[3]:


del(track_metadata_df_sub['track_id'])
del(track_metadata_df_sub['artist_mbid'])
track_metadata_df_sub = track_metadata_df_sub.drop_duplicates(['song_id'])
triplet_dataset_sub_song_merged = pd.merge(triplet_dataset_sub_song, track_metadata_df_sub, how='left', left_on='song', right_on='song_id')
triplet_dataset_sub_song_merged.rename(columns={'play_count':'listen_count'},inplace=True)


# In[4]:


del(triplet_dataset_sub_song_merged['song_id'])
del(triplet_dataset_sub_song_merged['artist_id'])
del(triplet_dataset_sub_song_merged['duration'])
del(triplet_dataset_sub_song_merged['artist_familiarity'])
del(triplet_dataset_sub_song_merged['artist_hotttnesss'])
del(triplet_dataset_sub_song_merged['track_7digitalid'])
del(triplet_dataset_sub_song_merged['shs_perf'])
del(triplet_dataset_sub_song_merged['shs_work'])


# In[5]:


triplet_dataset_sub_song_merged.head(n=10)


# # Some visualizations

# ## Most popular songs

# In[185]:


# popular_songs = triplet_dataset_sub_song_merged[['title','listen_count']].groupby('title').sum().reset_index()
# popular_songs_top_20 = popular_songs.sort_values('listen_count', ascending=False).head(n=20)
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
# objects = (list(popular_songs_top_20['title']))
# y_pos = np.arange(len(objects))
# performance = list(popular_songs_top_20['listen_count'])
#
# plt.bar(y_pos, performance, align='center', alpha=0.5)
# plt.xticks(y_pos, objects, rotation='vertical')
# plt.ylabel('Item count')
# plt.title('Most popular songs')
#
# plt.show()


# ## Most popular releases

# In[58]:


# popular_release = triplet_dataset_sub_song_merged[['release','listen_count']].groupby('release').sum().reset_index()
# popular_release_top_20 = popular_release.sort_values('listen_count', ascending=False).head(n=20)
#
# objects = (list(popular_release_top_20['release']))
# y_pos = np.arange(len(objects))
# performance = list(popular_release_top_20['listen_count'])
#
# plt.bar(y_pos, performance, align='center', alpha=0.5)
# plt.xticks(y_pos, objects, rotation='vertical')
# plt.ylabel('Item count')
# plt.title('Most popular Release')
#
# plt.show()


# ## Most popular artists

# In[62]:


# popular_artist = triplet_dataset_sub_song_merged[['artist_name','listen_count']].groupby('artist_name').sum().reset_index()
# popular_artist_top_20 = popular_artist.sort_values('listen_count', ascending=False).head(n=20)
#
# objects = (list(popular_artist_top_20['artist_name']))
# y_pos = np.arange(len(objects))
# performance = list(popular_artist_top_20['listen_count'])
#
# plt.bar(y_pos, performance, align='center', alpha=0.5)
# plt.xticks(y_pos, objects, rotation='vertical')
# plt.ylabel('Item count')
# plt.title('Most popular Artists')
#
# plt.show()


# ## Song count distribution

# In[64]:


# user_song_count_distribution = triplet_dataset_sub_song_merged[['user','title']].groupby('user').count().reset_index().sort_values(
# by='title',ascending = False)
# user_song_count_distribution.title.describe()


# In[71]:


# x = user_song_count_distribution.title
# n, bins, patches = plt.hist(x, 50, facecolor='green', alpha=0.75)
# plt.xlabel('Play Counts')
# plt.ylabel('Num of Users')
# plt.title(r'$\mathrm{Histogram\ of\ User\ Play\ Count\ Distribution}\ $')
# plt.grid(True)
# plt.show()


# # Recommendation Engines

# In[1]:


import Recommenders as Recommenders
from sklearn.model_selection import train_test_split


# ## Popularity based recommendations

# In[9]:

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
# x，y是原始的数据集。x_train,y_train 是原始数据集划分出来作为训练模型的，fit模型的时候用。
# x_test,y_test 这部分的数据不参与模型的训练，而是用于评价训练出来的模型好坏，score评分的时候用。
# test_size=0.2 测试集的划分比例 -- 测试集占总数据集的 0.4，剩下0.6就是训练集
# random_state=0 随机种子，如果随机种子一样，则随机生成的数据集是相同的
triplet_dataset_sub_song_merged_set = triplet_dataset_sub_song_merged
train_data, test_data = train_test_split(triplet_dataset_sub_song_merged_set, test_size = 0.40, random_state=0)


# In[10]:


train_data.head()


# In[11]:

# 1、基于热度的推荐引擎（排行榜推荐）
# 这种推荐引擎是最容易开发的。它的逻辑非常朴素：如果一样东西被很多人喜欢，那么推荐给更多的人一般来说也不会太坏。
def create_popularity_recommendation(train_data, user_id, item_id):
    #Get a count of user_ids for each unique song as recommendation score  获取每个独特歌曲的用户ID计数作为推荐分数
    train_data_grouped = train_data.groupby([item_id]).agg({user_id: 'count'}).reset_index()
    train_data_grouped.rename(columns = {user_id: 'score'},inplace=True)
    
    #Sort the songs based upon recommendation score  根据推荐分数对歌曲进行排序
    train_data_sort = train_data_grouped.sort_values(['score', item_id], ascending = [0,1])
    
    #Generate a recommendation rank based upon score  根据分数生成推荐等级
    train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')
        
    #Get the top 20 recommendations
    popularity_recommendations = train_data_sort.head(20)
    return popularity_recommendations


# In[82]:

# 关闭
# recommendations = create_popularity_recommendation(triplet_dataset_sub_song_merged,'user','title')


# In[83]:

# 关闭
# recommendations

# 2、基于内容相似的推荐
# ## Item similarity  based recommendations

# In[84]:

# 从所有歌曲中选取播放量前5000的歌曲
song_count_subset = song_count_df.head(n=5000)
user_subset = list(play_count_subset.user)
song_subset = list(song_count_subset.song)
# 关闭
# triplet_dataset_sub_song_merged_sub = triplet_dataset_sub_song_merged[triplet_dataset_sub_song_merged.song.isin(song_subset)]


# In[36]:

# 关闭
# triplet_dataset_sub_song_merged_sub.head()


# In[85]:

# 关闭
# train_data, test_data = train_test_split(triplet_dataset_sub_song_merged_sub, test_size = 0.30, random_state=0)
# #构造了很多变量和方法
# is_model = Recommenders.item_similarity_recommender_py()
# is_model.create(train_data, 'user', 'title')
# # 随机取一个用户
# user_id = list(train_data.user)[7]
# user_items = is_model.get_user_items(user_id)


# In[35]:

# 关闭
#Recommend songs for the user using personalized model
# is_model.recommend(user_id)


# ## Matrix factorization  based recommendations

# In[5]:
# 3、基于矩阵分解的推荐引擎
# 先计算 歌曲被当前用户播放量 / 当前用户总播放量   作为用户对歌曲打分
triplet_dataset_sub_song_merged_sum_df = triplet_dataset_sub_song_merged[['user','listen_count']].groupby('user').sum().reset_index()
triplet_dataset_sub_song_merged_sum_df.rename(columns={'listen_count':'total_listen_count'},inplace=True)
triplet_dataset_sub_song_merged = pd.merge(triplet_dataset_sub_song_merged,triplet_dataset_sub_song_merged_sum_df)
triplet_dataset_sub_song_merged['fractional_play_count'] = triplet_dataset_sub_song_merged['listen_count']/triplet_dataset_sub_song_merged['total_listen_count']
#
triplet_dataset_sub_song_merged.to_csv(path_or_buf=data_home+'triplet_dataset_sub_song_merged_test.csv', index=False)

# In[6]:


triplet_dataset_sub_song_merged[triplet_dataset_sub_song_merged.user =='d6589314c0a9bcbca4fee0c93b14bc402363afea'][['user','song','listen_count','fractional_play_count']].head()


# In[7]:


from scipy.sparse import coo_matrix
# 构造稀疏矩阵，节省空间
small_set = triplet_dataset_sub_song_merged
user_codes = small_set.user.drop_duplicates().reset_index()  # 只保留得分和用户信息
song_codes = small_set.song.drop_duplicates().reset_index()
user_codes.rename(columns={'index':'user_index'}, inplace=True)
song_codes.rename(columns={'index':'song_index'}, inplace=True)
song_codes['so_index_value'] = list(song_codes.index)  # 按照整数编码索引index
user_codes['us_index_value'] = list(user_codes.index)
small_set = pd.merge(small_set,song_codes,how='left')
small_set = pd.merge(small_set,user_codes,how='left')
mat_candidate = small_set[['us_index_value','so_index_value','fractional_play_count']]
data_array = mat_candidate.fractional_play_count.values
row_array = mat_candidate.us_index_value.values  # 行和列索引值
col_array = mat_candidate.so_index_value.values
#   (997, 1417)	0.0006779661016949153   997行1417列的值是0.0006779661016949153
data_sparse = coo_matrix((data_array, (row_array, col_array)),dtype=float)


# In[8]:


data_sparse


# In[9]:


user_codes[user_codes.user =='2a2f776cbac6df64d6cb505e7e834e01684673b6']


# In[10]:


import math as mt
from scipy.sparse.linalg import * #used for matrix multiplication
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix


# In[17]:


def compute_svd(urm, K):
    U, s, Vt = svds(urm, K)  # svd分解
    print("U-shape",U.shape)
    print("s-shape",s.shape)
    print("V-shape",Vt.shape)

    dim = (len(s), len(s))
    S = np.zeros(dim, dtype=np.float32)
    for i in range(0, len(s)):
        S[i,i] = mt.sqrt(s[i])
    # 转换成稀疏矩阵
    U = csc_matrix(U, dtype=np.float32)
    S = csc_matrix(S, dtype=np.float32)
    Vt = csc_matrix(Vt, dtype=np.float32)
    
    return U, S, Vt

def compute_estimated_matrix(urm, U, S, Vt, uTest, K, test):
    rightTerm = S*Vt 
    max_recommendation = 250
    estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)
    recomendRatings = np.zeros(shape=(MAX_UID,max_recommendation ), dtype=np.float16)
    for userTest in uTest:
        print("U[userTest,:]",U[userTest,:].shape)  # 获取当前测试的特征
        prod = U[userTest, :]*rightTerm   # 当前用户对所有歌曲的特征结果
        estimatedRatings[userTest, :] = prod.todense()
        recomendRatings[userTest, :] = (-estimatedRatings[userTest, :]).argsort()[:max_recommendation]
    return recomendRatings


# In[18]:


K=50  # 类似于中间特征值
urm = data_sparse   # 矩阵
MAX_PID = urm.shape[1]  # 用户个数
MAX_UID = urm.shape[0]   # 歌曲个数

U, S, Vt = compute_svd(urm, K)


# In[19]:

# 测试用户的索引值
uTest = [4,5,6,7,8,873,23]

uTest_recommended_items = compute_estimated_matrix(urm, U, S, Vt, uTest, K, True)


# In[20]:


for user in uTest:
    print("Recommendation for user with user id {}". format(user))
    rank_value = 1
    for i in uTest_recommended_items[user,0:10]:
        song_details = small_set[small_set.so_index_value == i].drop_duplicates('so_index_value')[['title','artist_name']]
        print("The number {} recommended song is {} BY {}".format(rank_value, list(song_details['title'])[0],list(song_details['artist_name'])[0]))
        rank_value+=1


# In[15]:


uTest = [27513]
#Get estimated rating for test user
print("Predictied ratings:")
uTest_recommended_items = compute_estimated_matrix(urm, U, S, Vt, uTest, K, True)


# In[16]:


for user in uTest:
    print("Recommendation for user with user id {}". format(user))
    rank_value = 1
    for i in uTest_recommended_items[user,0:10]:
        song_details = small_set[small_set.so_index_value == i].drop_duplicates('so_index_value')[['title','artist_name']]
        print("The number {} recommended song is {} BY {}".format(rank_value, list(song_details['title'])[0],list(song_details['artist_name'])[0]))
        rank_value+=1

