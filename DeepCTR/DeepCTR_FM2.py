from DeepCTR.deepctr_torch.layers.interaction import FM
import pandas as pd
import sqlite3

# 旧版本
# In[]

ratings= pd.read_csv("../data/metadata/user_item_rating_all_200w.csv")
track_all_200w = pd.read_csv("../data/metadata/track_all_200w.csv")
# track_all_200w.drop_duplicates(subset=['song'],keep='first',inplace=True)

# 筛选评分记录
# 字典user_playcounts记录每个用户的播放总量
# user_playcounts = {}
# for user, group in track_all_200w.groupby('user'):
#     user_playcounts[user] = group['play_count'].sum()
# temp_user = [user for user in user_playcounts.keys() if user_playcounts[user] > 100]
# temp_playcounts = [playcounts for user, playcounts in user_playcounts.items() if playcounts > 100]
# # track_all_200w = track_all_200w[track_all_200w.user.isin(temp_user)]
#
# ratings = ratings[ratings.user.isin(temp_user)]
# print('歌曲播放量大于100的用户数量占总体用户数量的比例为', str(round(len(temp_user)/len(user_playcounts), 4)*100)+'%')
# print('歌曲播放量大于100的用户产生的播放总量占总体播放总量的比例为', str(round(sum(temp_playcounts) / sum(user_playcounts.values())*100, 4))+'%')
# print('歌曲播放量大于100的用户产生的数据占总体数据的比例为', str(round(len(track_all_200w[track_all_200w.user.isin(temp_user)])/len(track_all_200w)*100, 4))+"%")
#
#
#
# # song_playcounts字典，记录每首歌的播放量
# song_playcounts = {}
# for song, group in track_all_200w.groupby('song'):
#     song_playcounts[song] = group['play_count'].sum()
# temp_song = [song for song in song_playcounts.keys() if song_playcounts[song] > 50]
# temp_playcounts = [playcounts for song, playcounts in song_playcounts.items() if playcounts > 50]
# # track_all_200w = track_all_200w[track_all_200w.song.isin(temp_song)]
#
# ratings = ratings[ratings.song.isin(temp_song)]
# print('播放量大于50的歌曲数量占总体歌曲数量的比例为', str(round(len(temp_song)/len(song_playcounts), 4)*100)+'%')
# print('播放量大于50的歌曲产生的播放总量占总体播放总量的比例为', str(round(sum(temp_playcounts) / sum(song_playcounts.values())*100, 4))+'%')
# print('播放量大于50的歌曲产生的数据占总体数据的比例为', str(round(len(track_all_200w[track_all_200w.song.isin(temp_song)])/len(track_all_200w)*100, 4))+"%")
#
# print("ratings筛选后:****************************************")
# track_all_200w.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)

conn = sqlite3.connect('../db/track_metadata.db')
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
cur.fetchall()
#
# # 获得数据的dataframe
track_metadata_df = pd.read_sql(con=conn, sql='select * from songs')
track_metadata_df = track_metadata_df[['track_id','duration']]

songs = pd.merge(track_all_200w,track_metadata_df,how='inner',on="track_id")
songs = songs[['song','artist_hotttnesss','year','duration']]
# 根据songID去重
songs.drop_duplicates(subset=['song'],keep='first',inplace=True)
# 根据year去除为空，去除为0
songs = songs.dropna(subset=['year'])
songs = songs[songs.year != 0]
print('根据songID去重，去0***********************')
songs.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)
# 去重artist_hotttnesss为空，为0
songs = songs.dropna(subset=['artist_hotttnesss'])
songs = songs[songs.artist_hotttnesss != 0]
print('去重artist_hotttnesss为空，为0***********************')
songs.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)

# 去重duration为空，为0
songs = songs.dropna(subset=['duration'])
songs = songs[songs.duration != 0]
print('duration***********************')
songs.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True)



# 评分中year为0的歌曲去除
song_with_year = [song for song in songs.song]
ratings = ratings[ratings.song.isin(song_with_year)]
del(song_with_year)

songs.head()

data = pd.merge(ratings,songs,how='inner',on="song")
n_users = data.user.unique().shape[0]
n_songs = data.song.unique().shape[0]

print('Number of users: {}'.format(n_users))
print('Number of songs: {}'.format(n_songs))
print('Sparsity: {:4.3f}%'.format(float(data.shape[0]) / float(n_users*n_songs) * 100))

del(songs)
del(ratings)
del(track_all_200w)
del(track_metadata_df)