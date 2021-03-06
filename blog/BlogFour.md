#### 《推荐系统实战（二）》音乐推荐系统（数据清洗、召回、排序）
https://blog.csdn.net/qq_30841655/article/details/107989560

前言
在本篇博客中，我们将从0搭建一个音乐推荐系统，其中的流程也可以用来搭建其他内容的推荐系统。

我们将整个过程分为三个部分，分别是

数据预处理
召回
排序
拿到原始数据集之后，我们需要对其进行处理，包括去重、重命名、去掉无用特征等等，最后形成较为简洁清晰的数据集。

有了数据集之后，我们进入系统的召回阶段。在这一阶段，我们从大量歌曲中选出少部分歌曲作为候选集，采用的方法有排行榜、协同过滤和矩阵分解。

通过召回阶段，我们得到歌曲的候选集，为了进一步筛选，我们采用GBDT+LR的ctr预估方法，对候选集进行进一步的打分和排序，最终选出得分最高的几个作为推荐结果输出。

Part 1. 数据集简介和预处理
Step 1.1. 数据集简介
我们的数据集是从网上的一个项目中获得的，这个项目由The Echonest和LABRosa一起完成。数据集主要是多年间外国音乐的量化特征，包含了百万用户对几十万首歌曲的播放记录（train_triplets.txt，2.9G）和这些歌曲的详细信息（triplets_metadata.db，700M）。

数据已经上传，免费下载；如果显示不免费，请评论区留言。

用户的播放记录数据集train_triplets.txt格式是这样的：用户 歌曲 播放次数，其中用户和歌曲都匿名。由于数据集很大，可以从.txt文件中选取200万条数据作为我们的数据集。

歌曲的详细信息数据集triplets_metadata.db则包括歌曲的发布时间、作者、作者热度等。

Step 1.2. 数据预处理
对于过大的文件，我们在读取的时候，可以改变其数据类型降低内存，比如

将float64转化为float32
将int64转化为int32
对object类型数据进行label编码
对.txt文件的处理
为了降低计算量以及提高准确性，我们过滤掉一些用户和歌曲。

对于用户而言，我们观察用户播放量的分布情况，如下，


从上图可以看到，有一大部分用户的歌曲播放量少于100。 少于100的歌曲播放量在持续几年的时间长度上来看是不正常的。 造成这种现象的原因，可能是这些用户不喜欢听歌，只是偶尔点开。

歌曲播放量大于100的用户占总体的40%，而正是这40%的用户，产生了80%的播放量，占据了总体数据的70%。 因此，我们可以直接将歌曲播放量少于100的用户过滤掉，而不影响整体数据。

我们观察歌曲的播放量的分布，如下，

我们观察到，大部分歌曲的播放量非常少，甚至不到50次！播放量大于50的歌曲数量，占总体数量的27%，而这27%的歌曲，产生的播放总量和数据总量都占90%以上！ 因此可以说，过滤掉这些播放量小于50的歌曲，对总体数据不会产生太大影响。

对.db文件的处理
.db文件里面有关于每一首歌曲的详细信息，所以我们在读取之后，要与.txt文件的数据进行合并。

数据清洗
对于合并之后的数据，我们主要去重以及去掉一些无用的特征，比如

track_id
artist_id
artist_mbid
duration
track_7digitalid
shs_perf
shs_work
处理之后，我们的数据是这样的，


Step 1.3. 可视化
我们简单来看一下数据。

第一张图：最受欢迎的歌手或者乐队


第二张图：最受欢迎的专辑

第三张图：最受欢迎的歌曲


Part 2. 召回阶段
对于系统的召回阶段，我们将给出如下三种推荐方式，分别是

基于排行榜的推荐
基于协同过滤的推荐
基于矩阵分解的推荐
Step 2.1. 基于排行榜的推荐
我们将每首歌听过的人数作为每首歌的打分。 这里之所以不将点击量作为打分，是因为一个人可能对一首歌多次点击，但这首歌其他人并不喜欢。

Step 2.2. 基于协同过滤的推荐
评分矩阵
协同过滤需要用户-物品评分矩阵。

考虑到不同用户听歌习惯不同，用户对一首歌的评分应该由这个用户本身的统计数据决定。因此，用户对某首歌的评分的计算公式如下，

该用户的最大歌曲点击量
当前歌曲点击量/最大歌曲点击量
评分=log(2 + 上述比值)
计算完评分之后，我们可以看一下整体评分的分布情况，如下


得到用户-物品评分矩阵之后，我们用surprise库中的knnbasic函数进行协同过滤。
#把x轴的刻度间隔设置为1，并存在变量里
x_major_locator=MultipleLocator(1)
#把y轴的刻度间隔设置为10，并存在变量里
#y_major_locator=MultipleLocator(10)
#ax为两条坐标轴的实例
ax=plt.gca()
#把x轴的主刻度设置为1的倍数
ax.xaxis.set_major_locator(x_major_locator)
#把y轴的主刻度设置为10的倍数
#ax.yaxis.set_major_locator(y_major_locator)
#把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
#plt.xlim(-0.5,11)
#plt.ylim(-5,110)

协同过滤主要有itemCF和userCF，想法类似，都是先计算相似度矩阵，根据相似度来推荐物品。我的博客里面有介绍，就不赘叙了。
[0.3522841911710546, 0.28709003703970487, 0.2794034690453446, 0.27656526733376197, 0.2762696817648763, 0.2763348196304486, 0.27621343112431757, 0.27618258150707126]
Step 2.3. 基于矩阵分解的推荐
矩阵分解同样采用上面的评分矩阵。

通过svd将稀疏的评分矩阵进行分解，预测得到用户对未播放音乐的评分。根据这个评分，系统做出推荐。

这里，我们用surprise库的svd方法来做分解。具体的原理在我博客里也有，有兴趣的可以去看看。

Part 3. 排序阶段
对于系统的排序阶段，我们通常是这样的，

以召回阶段的输出作为候选集
用CTR预估作为进一步的排序标准，从候选集中挑选少数几个作为推荐结果
这里，我们可以召回50首音乐，用GBDT+LR对这些音乐做ctr预估，给出评分排序，选出5首歌曲。

现在，仅仅用用户-物品评分是不够的，因为我们需要考虑特征之间的组合。为此，我们用之前的data数据。

为了保持数据集的正负平衡，我们令rating小于0.7的为0，也就是不喜欢，令rating大于0.7的为1，也就是喜欢。

然后我们用gbdt+lr做ctr预估，并用lr得到的概率作为权重，综合考虑候选集中歌曲的得分，得到最终的打分。

根据最终打分，我们选出最大的几首歌作为推荐结果。

Part 4. 结语
限于机器性能和时间所限，不能训练更多的数据，显然是未来可以提高的部分。在排序阶段，我们还可以用深度学习的相关算法，效果可能也不错。如果有更多的数据，比如像大众点评的结果查询结果，我们或许还可以做重排序。

附：github代码 https://github.com/wangxinRS/Recommendation-System-for-Songs


基于item、user、svd的knn方法( KFold(n_splits=5), KNNBasic(k=40) )：

Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.2760
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.2756
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.2762
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.2770
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.2758
k=40 itemCF的平均准确率rmse：0.27610
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.2723
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.2736
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.2710
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.2724
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.2745
k=40 userCF的平均准确率rmse：0.27276
RMSE: 0.2767
RMSE: 0.2747
RMSE: 0.2733
RMSE: 0.2747
RMSE: 0.2743
k=40 SVD的平均准确率rmse：0.27473
