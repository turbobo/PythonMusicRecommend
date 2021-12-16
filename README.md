### Python实现音乐推荐

Get more information about the Millionsong project from https://labrosa.ee.columbia.edu/millionsong/
Refer to Chapter 10: Section 'The Million Song Dataset Taste Profile' for more details

train_triplets.txt 原数据集：包含用户-歌曲-播放次数  [user, song, play_count]
数据来源：
The Echo Nest Taste Profile Subset: Echo Nest提供了可以与MSD关联的user-song-play count数据集，包含100万用户对38万首歌曲的4800万播放记录。
Get the data from http://labrosa.ee.columbia.edu/millionsong/sites/default/files/challenge/train_triplets.txt.zip

user_playcount_df.csv  每个用户播放歌曲的总次数  1019319  101w

song_playcount_df.csv  每首歌被播放的总次数   384547  38w

triplet_dataset_sub_song.csv   user_playcount_df播放前10w用户+song_playcount_df  播放前3w的歌曲筛选后的原数据集的子集

track_metadata.db  音乐详细信息

track_metadata_df_sub.csv  在triplet_dataset_sub_song中的歌曲信息子集 --- 修改参数后需要重新计算

lastfm_unique_tags.txt   每个tag的曲目数

track_tag_merge.txt  505216个曲目有标签


搜索“改进”

###参考博客
https://blog.csdn.net/kepengs/article/details/87621178


音乐推荐引擎
 

数据集
百万歌曲数据库
百万歌曲数据量可以在https://labrosa.ee.columbia.edu/millionsong/ 上下载。原始的数据包含了多年间上百万首歌曲的量化音频特征。它实际上是The Echonest和LABRosa的一个合作项目。
这里我们不会使用整个数据集，而只会使用它们中的一部分。
基于这个数据库，还衍生出了一些其他的数据集。其中一个就是The Echonest喜好画像子集。这个数据包含了匿名用户的歌曲播放次数的记录。这个数据集即使只是百万歌曲数据库的一个子集，但它的数据量也非常庞大，因为它包含了4800万行三元组信息：


这个数据大概包含了100万用户对384,000首歌的播放记录。
大家可以通过http://labrosa.ee.columbia.edu/millionsong/sites/default/files/challenge/train_triplets.txt.zip来下载。这个压缩文件大约500MB，解压后大约3.5GB。

数据探索
加载&裁剪数据
对于我们单机工作而言，这个数据太大了。但是如果是商用服务器，即使是单台机器，它能处理的数据量也要比这大得多，更不用说如果拥有集群计算能力的大型公司了。
不过，对于我们在现实工作中，我们也是常常从大数据量中抽取一些数据来在单机上进行分析、建模，这样做主要是数据量小的时候各种操作都非常快，同时也能验证我们想要做的事情是不是可行。
所以，在这里，我们也需要把数据进行一定的裁剪：

In [1]:

import pandas as pd
import numpy as np
import time
import sqlite3

import datetime
import math
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


%matplotlib inline

plt.rcParams['font.sans-serif']=['SimHei']     #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False       #用来正常显示负号
In [2]:

triplet_dataset = pd.read_csv(filepath_or_buffer='./data/train_triplets.txt', 
                              nrows=10000,sep='\t', header=None, 
                              names=['user','song','play_count'])     # 数据本身没有表头
In [3]:

triplet_dataset.head()
Out[3]:

 	user	song	play_count
0	b80344d063b5ccb3212f76538f3d9e43d87dca9e	SOAKIMP12A8C130995	1
1	b80344d063b5ccb3212f76538f3d9e43d87dca9e	SOAPDEY12A81C210A9	1
2	b80344d063b5ccb3212f76538f3d9e43d87dca9e	SOBBMDR12A8C13253B	2
3	b80344d063b5ccb3212f76538f3d9e43d87dca9e	SOBFNSP12AF72A0E22	1
4	b80344d063b5ccb3212f76538f3d9e43d87dca9e	SOBFOVM12A58A7D494	1
对于这样规模大小的数据集，我们首先要做的是有多少用户(或者歌曲)是我们应该要考虑的。在原始数据集中，有大约100万的用户，但是这里面是不是所有用户我们都需要纳入考虑呢？比如说，如果20%的用户的歌曲播放了占了80%的总体播放量，那么其实我们只需要考虑这20%用户就差不多了。
一般来说，我们统计一下播放量的累积求和就可以知道多少用户占了80%的总体播放量。不过鉴于数据量如此之大，pandas提供的累积求和功能会出问题。所以我们必须自己一行行地读取这个文件，一部分一部分地来完成这项工作：

In [4]:

output_dict = {}
with open('./data/train_triplets.txt') as f:
    for line_number, line in enumerate(f):
        user = line.split('\t')[0]
        play_count = int(line.split('\t')[2])
        if user in output_dict:
            play_count +=output_dict[user]
            output_dict.update({user:play_count})
        output_dict.update({user:play_count})
output_list = [{'user':k,'play_count':v} for k,v in output_dict.items()]
play_count_df = pd.DataFrame(output_list)
play_count_df = play_count_df.sort_values(by = 'play_count', ascending = False)

play_count_df.to_csv(path_or_buf='./data/user_playcount_df.csv', index = False)
In [5]:

play_count_df = pd.read_csv('./data/user_playcount_df.csv')
play_count_df.head()
Out[5]:

 	play_count	user
0	13132	093cb74eb3c517c5179ae24caf0ebec51b24d2a2
1	9884	119b7c88d58d0c6eb051365c103da5caf817bea6
2	8210	3fa44653315697f42410a30cb766a4eb102080bb
3	7015	a2679496cd0af9779a92a13ff7c6af5c81ea8c7b
4	6494	d7d2d888ae04d16e994d6964214a1de81392ee04
In [ ]:

output_dict = {}
with open('./data/train_triplets.txt') as f:
    for line_number, line in enumerate(f):
        song = line.split('\t')[1]
        play_count = int(line.split('\t')[2])
        if song in output_dict:
            play_count +=output_dict[song]
            output_dict.update({song:play_count})
        output_dict.update({song:play_count})
output_list = [{'song':k,'play_count':v} for k,v in output_dict.items()]
song_count_df = pd.DataFrame(output_list)
song_count_df = song_count_df.sort_values(by = 'play_count', ascending = False)

song_count_df.to_csv(path_or_buf='./data/song_playcount_df.csv', index = False)
In [6]:

song_count_df = pd.read_csv(filepath_or_buffer='./data/song_playcount_df.csv')
song_count_df.head()
Out[6]:

 	play_count	song
0	726885	SOBONKR12A58A7A7E0
1	648239	SOAUWYT12A81C206F1
2	527893	SOSXLTC12AF72A7F54
3	425463	SOFRQTD12A81C233C0
4	389880	SOEGIYH12A6D4FC0E3
有了这两份数据，我们首要的就是要找到前多少用户占了40%的总体播放量。这个"40%"是我们随机选的一个值，大家在实际的工作中可以自己选择这个数值，重点是控制数据集的大小。当然，如果有高效的Presto(支持HiveQL，但纯内存计算)集群的话，在整体数据集上统计这样的数据也会很快。
就我们这个数据集，大约前100,000用户的播放量占据了总体的40%。

In [7]:

total_play_count = sum(song_count_df.play_count)
print (float(play_count_df.head(n=100000).play_count.sum())/total_play_count)*100

play_count_subset = play_count_df.head(n=100000)
40.8807280501
同样的，我们发现大约30,000首歌占据了总体80%的播放量。这个信息就很有价值：10%的歌曲占据了80%的播放量。
那么，通过这样一些条件，我们就可以从原始的数据集中抽取出最具代表性的数据出来，从而使得需要处理的数据量在一个可控的范围内。

In [8]:

print (float(song_count_df.head(n=30000).play_count.sum())/total_play_count)*100

song_count_subset = song_count_df.head(n=30000)
78.3931536665
In [9]:

# 目标用户集和目标歌曲集
user_subset = list(play_count_subset.user)
song_subset = list(song_count_subset.song)
In [11]:

triplet_dataset = pd.read_csv(filepath_or_buffer='./data/train_triplets.txt',sep='\t', 
                              header=None, names=['user','song','play_count'])

# 抽取目标用户
triplet_dataset_sub = triplet_dataset[triplet_dataset.user.isin(user_subset) ]
del(triplet_dataset)

# 过滤非目标歌曲
triplet_dataset_sub_song = triplet_dataset_sub[triplet_dataset_sub.song.isin(song_subset)]
del(triplet_dataset_sub)

triplet_dataset_sub_song.to_csv('./data/triplet_dataset_sub_song.csv', index=False)
In [12]:

triplet_dataset_sub_song = pd.read_csv(filepath_or_buffer='./data/triplet_dataset_sub_song.csv')
In [13]:

triplet_dataset_sub_song.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10775200 entries, 0 to 10775199
Data columns (total 3 columns):
user          object
song          object
play_count    int64
dtypes: int64(1), object(2)
memory usage: 246.6+ MB
额外信息
我们前面加载的数据仅仅是三元组数据，我们既不知道歌曲的名称，也不知道歌手的名字，连专辑的名字都不知道。不过这份数据集其实也额外提供了这些歌曲相关的其他信息，比如歌曲名称、演唱者的名称、专辑名称等等。这份数据以SQLite数据库文件形式提供。原始的下载链接为：http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/track_metadata.db

In [14]:

conn = sqlite3.connect('./data/track_metadata.db')
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
cur.fetchall()
Out[14]:

[(u'songs',)]
In [15]:

track_metadata_df = pd.read_sql(con=conn, sql='select * from songs')
track_metadata_df_sub = track_metadata_df[track_metadata_df.song_id.isin(song_subset)]
In [16]:

track_metadata_df_sub.shape
Out[16]:

(30447, 14)
In [17]:

track_metadata_df_sub.head()
Out[17]:

 	track_id	title	song_id	release	artist_id	artist_mbid	artist_name	duration	artist_familiarity	artist_hotttnesss	year	track_7digitalid	shs_perf	shs_work
115	TRMMGCB128E079651D	Get Along (Feat: Pace Won) (Instrumental)	SOHNWIM12A67ADF7D9	Charango	ARU3C671187FB3F71B	067102ea-9519-4622-9077-57ca4164cfbb	Morcheeba	227.47383	0.819087	0.533117	2002	185967	-1	0
123	TRMMGTX128F92FB4D9	Viejo	SOECFIW12A8C144546	Caraluna	ARPAAPH1187FB3601B	f69d655c-ffd6-4bee-8c2a-3086b2be2fc6	Bacilos	307.51302	0.595554	0.400705	0	6825058	-1	0
145	TRMMGDP128F933E59A	I Say A Little Prayer	SOGWEOB12AB018A4D0	The Legendary Hi Records Albums_ Volume 3: Ful...	ARNNRN31187B9AE7B7	fb7272ba-f130-4f0a-934d-6eeea4c18c9a	Al Green	133.58975	0.779490	0.599210	1978	5211723	-1	11898
172	TRMMHBF12903CF6E59	At the Ball_ That's All	SOJGCRL12A8C144187	Best of Laurel & Hardy - The Lonesome Pine	AR1FEUF1187B9AF3E3	4a8ae4fd-ad6f-4912-851f-093f12ee3572	Laurel & Hardy	123.71546	0.438709	0.307120	0	8645877	-1	0
191	TRMMHKG12903CDB1B5	Black Gold	SOHNFBA12AB018CD1D	Total Life Forever	ARVXV1J1187FB5BF88	6a65d878-fcd0-42cf-aff9-ca1d636a8bcc	Foals	386.32444	0.842578	0.514523	2010	9007438	-1	0
In [18]:

# merge数据
del(track_metadata_df_sub['track_id'])
del(track_metadata_df_sub['artist_mbid'])
track_metadata_df_sub = track_metadata_df_sub.drop_duplicates(['song_id'])
triplet_dataset_sub_song_merged = pd.merge(triplet_dataset_sub_song, track_metadata_df_sub, how='left', left_on='song', right_on='song_id')
triplet_dataset_sub_song_merged.rename(columns={'play_count':'listen_count'},inplace=True)
In [19]:

# 删除无用字段
del(triplet_dataset_sub_song_merged['song_id'])
del(triplet_dataset_sub_song_merged['artist_id'])
del(triplet_dataset_sub_song_merged['duration'])
del(triplet_dataset_sub_song_merged['artist_familiarity'])
del(triplet_dataset_sub_song_merged['artist_hotttnesss'])
del(triplet_dataset_sub_song_merged['track_7digitalid'])
del(triplet_dataset_sub_song_merged['shs_perf'])
del(triplet_dataset_sub_song_merged['shs_work'])
In [20]:

triplet_dataset_sub_song_merged.head()
Out[20]:

 	user	song	listen_count	title	release	artist_name	year
0	d6589314c0a9bcbca4fee0c93b14bc402363afea	SOADQPP12A67020C82	12	You And Me Jesus	Tribute To Jake Hess	Jake Hess	2004
1	d6589314c0a9bcbca4fee0c93b14bc402363afea	SOAFTRR12AF72A8D4D	1	Harder Better Faster Stronger	Discovery	Daft Punk	2007
2	d6589314c0a9bcbca4fee0c93b14bc402363afea	SOANQFY12AB0183239	1	Uprising	Uprising	Muse	0
3	d6589314c0a9bcbca4fee0c93b14bc402363afea	SOAYATB12A6701FD50	1	Breakfast At Tiffany's	Home	Deep Blue Something	1993
4	d6589314c0a9bcbca4fee0c93b14bc402363afea	SOBOAFP12A8C131F36	7	Lucky (Album Version)	We Sing. We Dance. We Steal Things.	Jason Mraz & Colbie Caillat	0
In [78]:

# 为后面重复使用
triplet_dataset_sub_song_merged.to_csv('./data/triplet_dataset_sub_song_merged.csv',encoding='utf-8', index=False)
最流行的歌曲
In [28]:

popular_songs = triplet_dataset_sub_song_merged[['title','listen_count']].groupby('title').sum().reset_index()
popular_songs_top_20 = popular_songs.sort_values('listen_count', ascending=False).head(n=20)
 
objects = (list(popular_songs_top_20['title']))
y_pos = np.arange(len(objects))
performance = list(popular_songs_top_20['listen_count'])

plt.figure(figsize=(16,8)) 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation='vertical',fontsize=12)
plt.ylabel(u'播放次数')
plt.title(u'最流行歌曲')
 
plt.show()


In [26]:

popular_release = triplet_dataset_sub_song_merged[['release','listen_count']].groupby('release').sum().reset_index()
popular_release_top_20 = popular_release.sort_values('listen_count', ascending=False).head(n=20)

objects = (list(popular_release_top_20['release']))
y_pos = np.arange(len(objects))
performance = list(popular_release_top_20['listen_count'])
 
plt.figure(figsize=(16,8)) 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation='vertical',fontsize=12)
plt.ylabel(u'播放次数')
plt.title(u'最流行专辑')
 
plt.show()


最流行歌手
In [29]:

popular_artist = triplet_dataset_sub_song_merged[['artist_name','listen_count']].groupby('artist_name').sum().reset_index()
popular_artist_top_20 = popular_artist.sort_values('listen_count', ascending=False).head(n=20)

objects = (list(popular_artist_top_20['artist_name']))
y_pos = np.arange(len(objects))
performance = list(popular_artist_top_20['listen_count'])
 
plt.figure(figsize=(16,8)) 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation='vertical',fontsize=12)
plt.ylabel(u'播放次数')
plt.title(u'最流行歌手')
 
plt.show()


不过如果大家对这些音乐熟悉的话，可能会发现，虽然酷玩乐队(Coldplay)是最流行的乐队，但最热门的单曲中却没有他们的单曲。
如果仔细研究一下的话，会发现他们每首单曲的播放量都很平均，因此他们的总播放量可以排名第一，但每首单曲都没有进前20。

用户单曲分布
In [30]:

# 这里我们使用的是`count`，而不是`sum`，所以得到的结果是用户听过的单曲数
user_song_count_distribution = triplet_dataset_sub_song_merged[['user','title']].groupby('user').count().reset_index().sort_values(
by='title',ascending = False)
user_song_count_distribution.title.describe()
Out[30]:

count    99996.000000
mean       107.756310
std         79.737279
min          1.000000
25%         53.000000
50%         89.000000
75%        141.000000
max       1189.000000
Name: title, dtype: float64
In [34]:

x = user_song_count_distribution.title
plt.figure(figsize=(12,6))
n, bins, patches = plt.hist(x, 50, facecolor='green', alpha=0.75)
plt.xlabel(u'播放的单曲数')
plt.ylabel(u'用户量')
plt.grid(True)


我们在这个数据集上还可以进行更多的可视化操作，比如按照发布年份来分析、分析一下歌手专辑的流行度之类的。
相信大家现在已经有了足够的能力来对这些数据进行各种可视化操作，得到更多有意思的信息。

推荐引擎
推荐引擎要做的事情其实已经很明显了：推荐！
推荐的办法有很多，最长被讨论的有如下三种：

基于用户的推荐引擎
这种推荐引擎中，用户是最为重要的实体。它的基本逻辑是寻找用户间的相似性，然后以此作为推荐的基础。
基于内容的推荐引擎
在这种引擎中，很自然，内容是最为重要的实体，比如在我们这个案例中，歌曲就是核心。这种算法会去寻找内容的特征，然后建立内容间的相似性，基于这些相似性再做推荐
混合推荐引擎
这种其实也被称为协同过滤。
我们接下来的代码中会引用https://github.com/llSourcell中的代码。

基于热度的推荐引擎
这种推荐引擎是最容易开发的。它的逻辑非常朴素：如果一样东西被很多人喜欢，那么推荐给更多的人一般来说也不会太坏。

In [36]:

import Recommenders as Recommenders                       # 改编自https://github.com/llSourcell
from sklearn.model_selection import train_test_split
In [41]:

train_data, test_data = train_test_split(triplet_dataset_sub_song_merged, test_size = 0.40, random_state=0)
In [42]:

train_data.head()
Out[42]:

 	user	song	listen_count	title	release	artist_name	year
8742296	8272a3530646a31ef5e49ea894f928d0d6b9b31b	SOBTVDE12AF72A3DE5	1	Wish You Were Here	Morning View	Incubus	2001
4911823	74d54aded8585b89ef5e3d86f73bf4ce15a46e44	SOBBCWG12AF72AB9CB	1	Brothers	One Life Stand	Hot Chip	2010
5503975	a85cbab8153c5d9ef3dc40496602f2f6aa500acb	SOWYYUQ12A6701D68D	3	It's My Life	Crush	Bon Jovi	2000
7775708	6d24ea4af5d394408f2dbcc977bbb29d356e000d	SOXNFHG12A8C135C55	2	Drop	Labcabincalifornia	The Pharcyde	1995
3343780	3931fe199c4c42920ed84d72f57196d6c6046878	SOUGACV12A6D4F84E0	1	Mysteries	Show Your Bones	Yeah Yeah Yeahs	2006
In [43]:

def create_popularity_recommendation(train_data, user_id, item_id):
    train_data_grouped = train_data.groupby([item_id]).agg({user_id: 'count'}).reset_index()
    train_data_grouped.rename(columns = {user_id: 'score'},inplace=True)
    train_data_sort = train_data_grouped.sort_values(['score', item_id], ascending = [0,1])
    train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')
    popularity_recommendations = train_data_sort.head(20)
    return popularity_recommendations
In [44]:

recommendations = create_popularity_recommendation(triplet_dataset_sub_song_merged,'user','title')
recommendations
Out[44]:

 	title	score	Rank
19580	Sehr kosmisch	18628	1.0
5780	Dog Days Are Over (Radio Edit)	17638	2.0
27314	You're The One	16083	3.0
19542	Secrets	15136	4.0
18636	Revelry	14943	5.0
25070	Undo	14681	6.0
7531	Fireflies	13084	7.0
9641	Hey_ Soul Sister	12996	8.0
25216	Use Somebody	12791	9.0
9922	Horn Concerto No. 4 in E flat K495: II. Romanc...	12343	10.0
24291	Tive Sim	11829	11.0
3629	Canada	11592	12.0
23468	The Scientist	11538	13.0
4194	Clocks	11360	14.0
12136	Just Dance	11061	15.0
26974	Yellow	10922	16.0
16438	OMG	10818	17.0
9845	Home	10513	18.0
3296	Bulletproof	10381	19.0
4760	Creep (Explicit)	10242	20.0
基于内容相似的推荐
刚才我们开发了一个最简单的热榜推荐。现在我们来稍微开发一个更复杂一点的算法。
我们要开发的这个算法是基于计算歌曲相似度的。我们这里采用的相似度也很简单：



那么向用户k推荐歌曲的话，我们要做的是：

找出用户k听过的歌曲
针对他听过的每首歌计算一下跟所有歌曲的相似度
以相似度为准，找出相似度最高的歌曲并向用户推荐
显然，这个算法的第2条是计算密集型的任务，当歌曲数目较多时，显然计算量非常大。所以这里我们再一次缩减曲库：

<img width="700" alt="无标题" src="https://user-images.githubusercontent.com/46430934/143232789-b6b2059c-a1e6-4a90-9e7b-3bfa6ac43e8d.png">


In [45]:

song_count_subset = song_count_df.head(n=5000) # 选择最流行的5000首歌
user_subset = list(play_count_subset.user)
song_subset = list(song_count_subset.song)
triplet_dataset_sub_song_merged_sub = triplet_dataset_sub_song_merged[triplet_dataset_sub_song_merged.song.isin(song_subset)]
In [46]:

triplet_dataset_sub_song_merged_sub.head()
Out[46]:

 	user	song	listen_count	title	release	artist_name	year
0	d6589314c0a9bcbca4fee0c93b14bc402363afea	SOADQPP12A67020C82	12	You And Me Jesus	Tribute To Jake Hess	Jake Hess	2004
1	d6589314c0a9bcbca4fee0c93b14bc402363afea	SOAFTRR12AF72A8D4D	1	Harder Better Faster Stronger	Discovery	Daft Punk	2007
2	d6589314c0a9bcbca4fee0c93b14bc402363afea	SOANQFY12AB0183239	1	Uprising	Uprising	Muse	0
3	d6589314c0a9bcbca4fee0c93b14bc402363afea	SOAYATB12A6701FD50	1	Breakfast At Tiffany's	Home	Deep Blue Something	1993
4	d6589314c0a9bcbca4fee0c93b14bc402363afea	SOBOAFP12A8C131F36	7	Lucky (Album Version)	We Sing. We Dance. We Steal Things.	Jason Mraz & Colbie Caillat	0
In [47]:

train_data, test_data = train_test_split(triplet_dataset_sub_song_merged_sub, test_size = 0.30, random_state=0)
is_model = Recommenders.item_similarity_recommender_py()
is_model.create(train_data, 'user', 'title')
In [45]:

# 向用户推荐，即使5000篇，计算量也不小，大约需要1小时
user_id = list(train_data.user)[7]
user_items = is_model.get_user_items(user_id)
is_model.recommend(user_id)
No. of unique songs for the user: 82
no. of unique songs in the training set: 4879
Non zero values in cooccurence_matrix :378241
Out[45]:

 	user_id	song	score	rank
0	8ffa9a13c6fa5a3c04a95c449b148183dd51ebb7	Halo	0.046176	1
1	8ffa9a13c6fa5a3c04a95c449b148183dd51ebb7	Use Somebody	0.045396	2
2	8ffa9a13c6fa5a3c04a95c449b148183dd51ebb7	Secrets	0.043963	3
3	8ffa9a13c6fa5a3c04a95c449b148183dd51ebb7	I Kissed A Girl	0.043809	4
4	8ffa9a13c6fa5a3c04a95c449b148183dd51ebb7	Marry Me	0.043104	5
5	8ffa9a13c6fa5a3c04a95c449b148183dd51ebb7	The Only Exception (Album Version)	0.042511	6
6	8ffa9a13c6fa5a3c04a95c449b148183dd51ebb7	Fireflies	0.042496	7
7	8ffa9a13c6fa5a3c04a95c449b148183dd51ebb7	Billionaire [feat. Bruno Mars] (Explicit Albu...	0.042447	8
8	8ffa9a13c6fa5a3c04a95c449b148183dd51ebb7	Drop The World	0.042319	9
9	8ffa9a13c6fa5a3c04a95c449b148183dd51ebb7	Clocks	0.042271	10
注意到，我们这里仅仅将歌曲的听众作为特征，并没有使用任何歌曲自身的特征，而实际上这些特征都可以用来定义歌曲之间的相似度。
在现实的工业界场景中，相似度的衡量其实都是包含了非常多的各种各样的特征。

基于矩阵分解的推荐引擎
我们通过迭代的方式来求得内容的特征矩阵  和用户对这些特征兴趣的矩阵  。

 	张三(1)	李四(2)	王二(3)	麻子(4)
泰坦尼克号	5	5	0	0
乱世佳人	5	?	?	0
罗马假日	?	4	0	?
无间道	    0	0	5	4
指环王	    0	0	5	?


预测评分：



所以，

既然得到了这个式子，那么我们其实可以利用线性代数的知识来直接求解，而不去迭代的来求解和 。
当然了，考虑到矩阵分解的计算复杂度，我们在实际应用中其实更倾向于在理论课上讨论的迭代式的求解方式。

这里我们作为扩展的内容，使用矩阵分解直接来试试。 对我们而言，我们目前所知道的矩阵分解其实只有在PCA降维的时候简单学习到的 SVD 分解。
如果我们还记得使用 S 矩阵的前 K 个元素来挑选最重要的投影方向的话，我们其实也可以理解前 K 个元素对应的也是最重要的隐层特征。
所以，我们可以借助 SVD 来构造这里的两个分解。那么基本的步骤是：

将用户播放矩阵进行分解，得到矩阵
选择  的前  个元素(对角线)
计算  的平方根得到 
分别计算  和  作为用户喜好矩阵和内容特征矩阵
因为内存限制的原因，在运行下面的代码前最好 "Restart"一下Kernel

In [2]:

triplet_dataset_sub_song_merged = pd.read_csv('./data/triplet_dataset_sub_song_merged.csv',encoding='utf-8')
In [3]:

# 因为我们没有用户评分，只有用户播放的记录，因此我们使用用户播百分比作为评分
triplet_dataset_sub_song_merged_sum_df = triplet_dataset_sub_song_merged[['user','listen_count']].groupby('user').sum().reset_index()
triplet_dataset_sub_song_merged_sum_df.rename(columns={'listen_count':'total_listen_count'},inplace=True)
triplet_dataset_sub_song_merged = pd.merge(triplet_dataset_sub_song_merged,triplet_dataset_sub_song_merged_sum_df)
triplet_dataset_sub_song_merged['fractional_play_count'] = triplet_dataset_sub_song_merged['listen_count']/triplet_dataset_sub_song_merged['total_listen_count']
In [3]:

triplet_dataset_sub_song_merged[triplet_dataset_sub_song_merged.user =='d6589314c0a9bcbca4fee0c93b14bc402363afea'][['user','song','listen_count','fractional_play_count']].head()
Out[3]:

 	user	song	listen_count	fractional_play_count
0	d6589314c0a9bcbca4fee0c93b14bc402363afea	SOADQPP12A67020C82	12	0.036474
1	d6589314c0a9bcbca4fee0c93b14bc402363afea	SOAFTRR12AF72A8D4D	1	0.003040
2	d6589314c0a9bcbca4fee0c93b14bc402363afea	SOANQFY12AB0183239	1	0.003040
3	d6589314c0a9bcbca4fee0c93b14bc402363afea	SOAYATB12A6701FD50	1	0.003040
4	d6589314c0a9bcbca4fee0c93b14bc402363afea	SOBOAFP12A8C131F36	7	0.021277
In [4]:

# 准备好 用户-歌曲 "评分"矩阵
from scipy.sparse import coo_matrix

small_set = triplet_dataset_sub_song_merged
user_codes = small_set.user.drop_duplicates().reset_index()  
song_codes = small_set.song.drop_duplicates().reset_index()
user_codes.rename(columns={'index':'user_index'}, inplace=True)
song_codes.rename(columns={'index':'song_index'}, inplace=True)
song_codes['so_index_value'] = list(song_codes.index)
user_codes['us_index_value'] = list(user_codes.index)
small_set = pd.merge(small_set,song_codes,how='left')
small_set = pd.merge(small_set,user_codes,how='left')
mat_candidate = small_set[['us_index_value','so_index_value','fractional_play_count']]
data_array = mat_candidate.fractional_play_count.values
row_array = mat_candidate.us_index_value.values
col_array = mat_candidate.so_index_value.values

data_sparse = coo_matrix((data_array, (row_array, col_array)),dtype=float)
In [5]:

data_sparse
Out[5]:

<99996x30000 sparse matrix of type '<type 'numpy.float64'>'
	with 10775200 stored elements in COOrdinate format>
In [6]:

user_codes[user_codes.user =='2a2f776cbac6df64d6cb505e7e834e01684673b6']
Out[6]:

 	user_index	user	us_index_value
27514	2981481	2a2f776cbac6df64d6cb505e7e834e01684673b6	27514
In [7]:

import math as mt
from scipy.sparse.linalg import * #used for matrix multiplication
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
In [8]:

def compute_svd(urm, K):
    U, s, Vt = svds(urm, K)

    dim = (len(s), len(s))
    S = np.zeros(dim, dtype=np.float32)
    for i in range(0, len(s)):
        S[i,i] = mt.sqrt(s[i]) # 求平方根

    U = csc_matrix(U, dtype=np.float32)
    S = csc_matrix(S, dtype=np.float32)
    Vt = csc_matrix(Vt, dtype=np.float32)
    
    return U, S, Vt

def compute_estimated_matrix(urm, U, S, Vt, uTest, K):
    rightTerm = S*Vt 
    max_recommendation = 250
    estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)
    recomendRatings = np.zeros(shape=(MAX_UID,max_recommendation ), dtype=np.float16)
    for userTest in uTest:
        prod = U[userTest, :]*rightTerm
        estimatedRatings[userTest, :] = prod.todense()
        recomendRatings[userTest, :] = (-estimatedRatings[userTest, :]).argsort()[:max_recommendation]
    return recomendRatings
In [9]:

K=50
urm = data_sparse
MAX_PID = urm.shape[1]
MAX_UID = urm.shape[0]

U, S, Vt = compute_svd(urm, K)
In [10]:

uTest = [4,5,6,7,8,873,23]

uTest_recommended_items = compute_estimated_matrix(urm, U, S, Vt, uTest, K)
In [11]:

for user in uTest:
    print u"Recommendation for user with user id {}". format(user)
    rank_value = 1
    for i in uTest_recommended_items[user,0:10]:
        song_details = small_set[small_set.so_index_value == i].drop_duplicates('so_index_value')[['title','artist_name']]
        print u"The number {} recommended song is {} BY {}".format(rank_value, list(song_details['title'])[0],list(song_details['artist_name'])[0])
        rank_value+=1
Recommendation for user with user id 4
The number 1 recommended song is Fireflies BY Charttraxx Karaoke
The number 2 recommended song is Hey_ Soul Sister BY Train
The number 3 recommended song is OMG BY Usher featuring will.i.am
The number 4 recommended song is Lucky (Album Version) BY Jason Mraz & Colbie Caillat
The number 5 recommended song is Vanilla Twilight BY Owl City
The number 6 recommended song is Crumpshit BY Philippe Rochard
The number 7 recommended song is Billionaire [feat. Bruno Mars]  (Explicit Album Version) BY Travie McCoy
The number 8 recommended song is Love Story BY Taylor Swift
The number 9 recommended song is TULENLIEKKI BY M.A. Numminen
The number 10 recommended song is Use Somebody BY Kings Of Leon
Recommendation for user with user id 5
The number 1 recommended song is Sehr kosmisch BY Harmonia
The number 2 recommended song is Dog Days Are Over (Radio Edit) BY Florence + The Machine
The number 3 recommended song is Ain't Misbehavin BY Sam Cooke
The number 4 recommended song is Revelry BY Kings Of Leon
The number 5 recommended song is Undo BY Björk
The number 6 recommended song is Cosmic Love BY Florence + The Machine
The number 7 recommended song is Home BY Edward Sharpe & The Magnetic Zeros
The number 8 recommended song is You've Got The Love BY Florence + The Machine
The number 9 recommended song is Bring Me To Life BY Evanescence
The number 10 recommended song is Tighten Up BY The Black Keys
Recommendation for user with user id 6
The number 1 recommended song is Crumpshit BY Philippe Rochard
The number 2 recommended song is Marry Me BY Train
The number 3 recommended song is Hey_ Soul Sister BY Train
The number 4 recommended song is Lucky (Album Version) BY Jason Mraz & Colbie Caillat
The number 5 recommended song is One On One BY the bird and the bee
The number 6 recommended song is I Never Told You BY Colbie Caillat
The number 7 recommended song is Canada BY Five Iron Frenzy
The number 8 recommended song is Fireflies BY Charttraxx Karaoke
The number 9 recommended song is TULENLIEKKI BY M.A. Numminen
The number 10 recommended song is Bring Me To Life BY Evanescence
Recommendation for user with user id 7
The number 1 recommended song is Behind The Sea [Live In Chicago] BY Panic At The Disco
The number 2 recommended song is The City Is At War (Album Version) BY Cobra Starship
The number 3 recommended song is Dead Souls BY Nine Inch Nails
The number 4 recommended song is Una Confusion BY LU
The number 5 recommended song is Home BY Edward Sharpe & The Magnetic Zeros
The number 6 recommended song is Climbing Up The Walls BY Radiohead
The number 7 recommended song is Tighten Up BY The Black Keys
The number 8 recommended song is Tive Sim BY Cartola
The number 9 recommended song is West One (Shine On Me) BY The Ruts
The number 10 recommended song is Cosmic Love BY Florence + The Machine
Recommendation for user with user id 8
The number 1 recommended song is Undo BY Björk
The number 2 recommended song is Canada BY Five Iron Frenzy
The number 3 recommended song is Better To Reign In Hell BY Cradle Of Filth
The number 4 recommended song is Unite (2009 Digital Remaster) BY Beastie Boys
The number 5 recommended song is Behind The Sea [Live In Chicago] BY Panic At The Disco
The number 6 recommended song is Rockin' Around The Christmas Tree BY Brenda Lee
The number 7 recommended song is Tautou BY Brand New
The number 8 recommended song is Revelry BY Kings Of Leon
The number 9 recommended song is 16 Candles BY The Crests
The number 10 recommended song is Catch You Baby (Steve Pitron & Max Sanna Radio Edit) BY Lonnie Gordon
Recommendation for user with user id 873
The number 1 recommended song is The Scientist BY Coldplay
The number 2 recommended song is Yellow BY Coldplay
The number 3 recommended song is Clocks BY Coldplay
The number 4 recommended song is Fix You BY Coldplay
The number 5 recommended song is In My Place BY Coldplay
The number 6 recommended song is Shiver BY Coldplay
The number 7 recommended song is Speed Of Sound BY Coldplay
The number 8 recommended song is Creep (Explicit) BY Radiohead
The number 9 recommended song is Sparks BY Coldplay
The number 10 recommended song is Use Somebody BY Kings Of Leon
Recommendation for user with user id 23
The number 1 recommended song is Garden Of Eden BY Guns N' Roses
The number 2 recommended song is Don't Speak BY John Dahlbäck
The number 3 recommended song is Master Of Puppets BY Metallica
The number 4 recommended song is TULENLIEKKI BY M.A. Numminen
The number 5 recommended song is Bring Me To Life BY Evanescence
The number 6 recommended song is Kryptonite BY 3 Doors Down
The number 7 recommended song is Make Her Say BY Kid Cudi / Kanye West / Common
The number 8 recommended song is Night Village BY Deep Forest
The number 9 recommended song is Better To Reign In Hell BY Cradle Of Filth
The number 10 recommended song is Xanadu BY Olivia Newton-John;Electric Light Orchestra
开源推荐引擎库
我们这里简单地实现了一个基于矩阵分解的推荐引擎，虽然非常简单，但希望能给大家一个简明的认识。
当然，在python中也有一些开源的推荐引擎库：

scikit-surprise
lightfm
crab
rec_sys
...
