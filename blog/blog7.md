**Python实现基于标签的推荐系统**


标签：泛指分类、关键词、作者、标签等一切离散的内容特征

实现思路：

根据用户看过的内容列表，聚合统计出（用户、标签、频次）数据，即每个用户最喜欢的标签
根据用户看过的内容列表，聚合统计出（标签、内容、频次）数据，即每个标签最热的内容
对每个用户，根据最喜欢的标签取出标签的热榜/最新/人工推荐等列表，加权得到推荐列表
# 给这个用户作推荐
target_user_id = 1
1. 读取MovieLens数据集
import pandas as pd
import numpy as np
df = pd.merge(
    left=pd.read_csv("./datas/ml-latest-small/ratings.csv"),
    right=pd.read_csv("./datas/ml-latest-small/movies.csv"),
    left_on="movieId", right_on="movieId"
)
df.head()
userId	movieId	rating	timestamp	title	genres
0	1	1	4.0	964982703	Toy Story (1995)	Adventure|Animation|Children|Comedy|Fantasy
1	5	1	4.0	847434962	Toy Story (1995)	Adventure|Animation|Children|Comedy|Fantasy
2	7	1	4.5	1106635946	Toy Story (1995)	Adventure|Animation|Children|Comedy|Fantasy
3	15	1	2.5	1510577970	Toy Story (1995)	Adventure|Animation|Children|Comedy|Fantasy
4	17	1	4.5	1305696483	Toy Story (1995)	Adventure|Animation|Children|Comedy|Fantasy
2. 计算目标用户最喜欢的标签列表
df_target_user = df[df["userId"] == target_user_id]
df_target_user.head()
userId	movieId	rating	timestamp	title	genres
0	1	1	4.0	964982703	Toy Story (1995)	Adventure|Animation|Children|Comedy|Fantasy
215	1	3	4.0	964981247	Grumpier Old Men (1995)	Comedy|Romance
267	1	6	4.0	964982224	Heat (1995)	Action|Crime|Thriller
369	1	47	5.0	964983815	Seven (a.k.a. Se7en) (1995)	Mystery|Thriller
572	1	50	5.0	964982931	Usual Suspects, The (1995)	Crime|Mystery|Thriller
from collections import defaultdict
user_genres = defaultdict(int)
# 评分的累加，作为标签的分数
for index, row in df_target_user.iterrows():
    for genre in row["genres"].split("|"):
        user_genres[genre] += row["rating"]
sorted(user_genres.items(), key=lambda x: x[1], reverse=True)[:10]
[('Action', 389.0),
 ('Adventure', 373.0),
 ('Comedy', 355.0),
 ('Drama', 308.0),
 ('Thriller', 228.0),
 ('Fantasy', 202.0),
 ('Crime', 196.0),
 ('Children', 191.0),
 ('Sci-Fi', 169.0),
 ('Animation', 136.0)]
# 归一化
min_value = min(list(user_genres.values()))
max_value = max(list(user_genres.values()))
for genre, score in user_genres.items():
    user_genres[genre] = (score - min_value) / (max_value-min_value)
sorted(user_genres.items(), key=lambda x: x[1], reverse=True)[:10]
[('Action', 1.0),
 ('Adventure', 0.9583333333333334),
 ('Comedy', 0.9114583333333334),
 ('Drama', 0.7890625),
 ('Thriller', 0.5807291666666666),
 ('Fantasy', 0.5130208333333334),
 ('Crime', 0.4973958333333333),
 ('Children', 0.484375),
 ('Sci-Fi', 0.4270833333333333),
 ('Animation', 0.3411458333333333)]
3. 计算每个标签下的电影热榜
# 格式为：{genre: [movie, score]}
genre_hots = defaultdict(dict)
# 计算每个电影下的热榜
for index, row in df.iterrows():
    movie_id = row["movieId"]
    for genre in row["genres"].split("|"):
        if movie_id not in genre_hots[genre]:
            genre_hots[genre][movie_id] = 0.0
        genre_hots[genre][movie_id] += row["rating"]
sorted(genre_hots["Adventure"].items(), key=lambda x: x[1], reverse=True)[:10]
[(260, 1062.0),
 (480, 892.5),
 (1196, 889.5),
 (1, 843.0),
 (1198, 841.5),
 (4993, 813.0),
 (1210, 811.0),
 (150, 773.0),
 (7153, 762.0),
 (5952, 756.0)]
# 归一化
for genre in genre_hots:
    min_value = min(list(genre_hots[genre].values()))
    max_value = max(list(genre_hots[genre].values()))
    for movie_id, rating in genre_hots[genre].items():
        genre_hots[genre][movie_id] = (rating - min_value) / (max_value-min_value)
sorted(genre_hots["Adventure"].items(), key=lambda x: x[1], reverse=True)[:10]
[(260, 1.0),
 (480, 0.8403203014601979),
 (1196, 0.8374941121055111),
 (1, 0.7936881771078662),
 (1198, 0.7922750824305228),
 (4993, 0.7654262835609986),
 (1210, 0.7635421573245408),
 (150, 0.7277437588318417),
 (7153, 0.7173810645313236),
 (5952, 0.71172868582195)]
4. 计算用户喜爱的标签最热的10个电影
# {movie_id: score}
target_movies = defaultdict(float)
for user_genre, user_genre_score in user_genres.items():
    for movie, movie_score in genre_hots[user_genre].items():
        target_movies[movie] += user_genre_score * movie_score
sorted(target_movies.items(), key=lambda x: x[1], reverse=True)[:10]
[(1, 2.659877754856764),
 (296, 2.618536103626324),
 (480, 2.3001570533654525),
 (260, 2.2586328683834047),
 (7153, 2.2329593476964504),
 (4306, 2.207907832512942),
 (356, 2.2048431748010926),
 (2959, 2.072269210220652),
 (588, 2.0223837340176054),
 (2571, 1.9523546680900619)]
5. 拼接ID得到电影名称给出最终列表
df_target = pd.merge(
    left=pd.DataFrame(target_movies.items(), columns=["movieId", "score"]),
    right=pd.read_csv("./datas/ml-latest-small/movies.csv"),
    on="movieId"
)
df_target.sort_values(by="score", ascending=False).head(10)
movieId	score	title	genres
0	1	2.659878	Toy Story (1995)	Adventure|Animation|Children|Comedy|Fantasy
1840	296	2.618536	Pulp Fiction (1994)	Comedy|Crime|Drama|Thriller
6	480	2.300157	Jurassic Park (1993)	Action|Adventure|Sci-Fi|Thriller
3	260	2.258633	Star Wars: Episode IV - A New Hope (1977)	Action|Adventure|Sci-Fi
165	7153	2.232959	Lord of the Rings: The Return of the King, The...	Action|Adventure|Drama|Fantasy
148	4306	2.207908	Shrek (2001)	Adventure|Animation|Children|Comedy|Fantasy|Ro...
1842	356	2.204843	Forrest Gump (1994)	Comedy|Drama|Romance|War
5668	2959	2.072269	Fight Club (1999)	Action|Crime|Drama|Thriller
95	588	2.022384	Aladdin (1992)	Adventure|Animation|Children|Comedy|Musical
5663	2571	1.952355	Matrix, The (1999)	Action|Sci-Fi|Thriller
 