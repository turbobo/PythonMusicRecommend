import pandas as pd
import numpy as np
import torch

from collections import namedtuple
from itertools import chain
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
# %matplotlib inline

#%%
#@title 默认标题文本
#加载数据
path = "../data/metadata/"

# 这个数据大概包含了大约1019319用户对384547首歌的48373585条播放记录。
# data = pd.read_csv(path+'track_200w.txt',
#                 sep='\t', header=None,
#                 names=['user','song','play_count'])
data = pd.read_csv(path+'track_200w.csv')
data.info()
data.head()

data = data[['user','song','play_count','year','tags']]
# # label编码
# user_encoder = LabelEncoder()
# data['user'] = user_encoder.fit_transform(data['user'].values)

# song_encoder = LabelEncoder()
# data['song'] = song_encoder.fit_transform(data['song'].values)


# 数据类型转换
data.astype({'user': 'int32', 'song': 'int32', 'play_count': 'int32'})

# 当前内存结果
data.info()


# 用户的歌曲播放总量的分布
# 字典user_playcounts记录每个用户的播放总量
user_playcounts = {}
for user, group in data.groupby('user'):
    user_playcounts[user] = group['play_count'].sum()


# In[7]:


# 作图
# sns.displot(list(user_playcounts.values()), bins=5000, kde=False)
# plt.xlim(0, 500)
# plt.xlabel('play_count')
# plt.ylabel('nums of user')
# plt.show()

# 从上图可以看到，有一大部分用户的歌曲播放量少于100。
# 少于100的歌曲播放量在持续几年的时间长度上来看是不正常的。
# 造成这种现象的原因，可能是这些用户不喜欢听歌，只是偶尔点开。
# 对于这些用户，我们看看他们在总体数据上的占比情况。

# In[8]:


temp_user = [user for user in user_playcounts.keys() if user_playcounts[user] > 100]
temp_playcounts = [playcounts for user, playcounts in user_playcounts.items() if playcounts > 100]

# print('歌曲播放量大于100的用户数量占总体用户数量的比例为', str(round(len(temp_user)/len(user_playcounts), 4)*100)+'%')
# print('歌曲播放量大于100的用户产生的播放总量占总体播放总量的比例为', str(round(sum(temp_playcounts) / sum(user_playcounts.values())*100, 4))+'%')
# print('歌曲播放量大于100的用户产生的数据占总体数据的比例为', str(round(len(data[data.user.isin(temp_user)])/len(data)*100, 4))+"%")


# 通过上面的结果，我们可以看到，歌曲播放量大于100的用户占总体的40%，而正是这40%的用户，产生了80%的播放量，占据了总体数据的70%。
# 因此，我们可以直接将歌曲播放量少于100的用户过滤掉，而不影响整体数据。

# 过滤掉歌曲播放量少于100的用户的数据
data = data[data.user.isin(temp_user)]

data.info()
data.columns

# 类似的，我们挑选出具有一定播放量的歌曲。因为播放量太低的歌曲不但会增加计算复杂度，还会降低协同过滤的准确度。
# 我们首先看不同歌曲的播放量分布情况。

# In[10]:


# song_playcounts字典，记录每首歌的播放量
song_playcounts = {}
for song, group in data.groupby('song'):
    song_playcounts[song] = group['play_count'].sum()


# In[11]:


# 作图
# sns.displot(list(song_playcounts.values()), bins=10000, kde=False)
# plt.xlim(0, 200)
# plt.xlabel('play_count')
# plt.ylabel('nums of song')
# plt.show()


# 我们观察到，大部分歌曲的播放量非常少，甚至不到50次！这些歌曲完全无人问津，属于我们可以过滤掉的对象。

# In[12]:


temp_song = [song for song in song_playcounts.keys() if song_playcounts[song] > 50]
temp_playcounts = [playcounts for song, playcounts in song_playcounts.items() if playcounts > 50]

# print('播放量大于50的歌曲数量占总体歌曲数量的比例为', str(round(len(temp_song)/len(song_playcounts), 4)*100)+'%')
# print('播放量大于50的歌曲产生的播放总量占总体播放总量的比例为', str(round(sum(temp_playcounts) / sum(song_playcounts.values())*100, 4))+'%')
# print('播放量大于50的歌曲产生的数据占总体数据的比例为', str(round(len(data[data.song.isin(temp_song)])/len(data)*100, 4))+"%")
#

# 可以看到，播放量大于50的歌曲数量，占总体数量的27%，而这27%的歌曲，产生的播放总量和数据总量都占90%以上！
# 因此可以说，过滤掉这些播放量小于50的歌曲，对总体数据不会产生太大影响。

# In[13]:


# 过滤掉播放量小于50的歌曲
data = data[data.song.isin(temp_song)]
data.info()
data.columns

#评分
# 每首歌播放量占总播放量比例
# data['rating'] = data.apply(lambda x: (x.play_count / user_playcounts[x.user]), axis=1)
#
# # 得到用户-音乐评分矩阵
# user_item_rating = data[['user', 'song', 'rating']]
# user_item_rating.to_csv('./user_item_rating_2.csv', index=False)   # 写入文件



# 直接用播放次数代替评分
# ratings = pd.read_csv(path+'user_song_playcounts.csv')   #, sep=',', header=None, engine='python')
ratings = pd.read_csv(path+'user_item_rating_2.csv')   #, sep=',', header=None, engine='python')
ratings.columns = ['user','song','rating']
# ratings = ratings.sample(frac=0.1, axis=0)

# 数据类型转换
ratings.astype({'user': 'int32', 'song': 'int32', 'rating': 'int32'})
print('ratings.info()')
ratings.info()

# track
# track_sub = pd.read_csv(path+'track_200w.csv')
# track_sub.info()
# 随机取0.1
# track_sub = track_sub.sample(frac=0.1, axis=0)

#songs
# movies = pd.read_csv(path+'track_200w.csv', sep=',', header=None, engine='python')
# movies.columns = ['song','title','genres']
# songs = track_sub[['song','tags','year']]
# songs = track_sub[['song','year']]

# # year有为0的数据，用艺术家热度代替
# songs = track_sub[['song','year']]
# songs = songs.rename(columns={'year': 'song_year'})
# # 根据song去重
# songs.drop_duplicates(subset=['song'],keep='first',inplace=True)
# print('songs.info()')
# songs.info()
#
# #Users
# # users = pd.read_csv(path+'users.dat', sep='::', header=None, engine='python')
# # users.columns = ['userId','gender','age','occupation','zipCode']
# users = track_sub[['user']]
#
# # 用完后释放内存
# del(track_sub)


songs = data[['song','year','tags']]
songs = songs.rename(columns={'year': 'song_year'})
# 根据song去重
songs.drop_duplicates(subset=['song'],keep='first',inplace=True)
print('songs.info()')
songs.info()

#Users
# users = pd.read_csv(path+'users.dat', sep='::', header=None, engine='python')
# users.columns = ['userId','gender','age','occupation','zipCode']
# users = track_sub[['user']]
users = data[['user']]

# 用完后释放内存
del(data)

#数据分析
print('Duplicated rows in ratings file: ' + str(ratings.duplicated().sum()))

n_users = ratings.user.unique().shape[0]
n_songs = ratings.song.unique().shape[0]

print('Number of users: {}'.format(n_users))
print('Number of songs: {}'.format(n_songs))
print('Sparsity: {:4.3f}%'.format(float(ratings.shape[0]) / float(n_users*n_songs) * 100))

rating_song = pd.merge(ratings,songs,how='left',on="song")

final_df = pd.merge(rating_song,users,how='left',on='user')

final_df.info()

final_df.head()

final_df = final_df.drop_duplicates((['user', 'song']))
# songs.drop_duplicates(subset=['song'],keep='first',inplace=True)

print('根据user-song去重后：')
final_df.info()

final_df.head()

# 去除song_year为0的记录
# final_df = final_df[final_df.song_year != 0]
final_df = final_df.dropna(subset=['song_year'])
# final_df.dropna(axis=0,how='any')
final_df.info()
final_df.head()


#%%
# ********************************************************************************

'''
#加载数据
path = "data/metadata/"

#评分
# 把用户收听记录转为整数评分
# ratings = pd.read_csv(path+'user_item_rating.csv')   #, sep=',', header=None, engine='python')
ratings = pd.read_csv(path+'user_song_playcounts.csv')   #, sep=',', header=None, engine='python')
ratings.columns = ['user','song','rating']
ratings = ratings.sample(frac=0.1, axis=0)
# 数据类型转换
ratings.astype({'user': 'int32', 'song': 'int32', 'rating': 'int32'})
print('ratings.info()')
ratings.info()

# track
track_sub =  pd.read_csv(path+'track_200w.csv')
# 随机取0.1
# track_sub = track_sub.sample(frac=0.1, axis=0)
track_sub.info()
#songs
# movies = pd.read_csv(path+'track_200w.csv', sep=',', header=None, engine='python')
# movies.columns = ['song','title','genres']
# songs = track_sub[['song','tags','year']]
# 去除tag列
# songs = track_sub[['song','year']]
songs = track_sub[['song','artist_hotttnesss']]
songs = songs.rename(columns={'artist_hotttnesss': 'song_year'})
# 根据song去重
songs.drop_duplicates(subset=['song'],keep='first',inplace=True)
print('songs.info()')
songs.info()

#Users
# users = pd.read_csv(path+'users.dat', sep='::', header=None, engine='python')
# users.columns = ['userId','gender','age','occupation','zipCode']
users = track_sub[['user']]

# 用完后释放内存
del(track_sub)


#数据分析
print('Duplicated rows in ratings file: ' + str(ratings.duplicated().sum()))

n_users = ratings.user.unique().shape[0]
n_songs = ratings.song.unique().shape[0]

print('Number of users: {}'.format(n_users))
print('Number of songs: {}'.format(n_songs))
print('Sparsity: {:4.3f}%'.format(float(ratings.shape[0]) / float(n_users*n_songs) * 100))


## 数据预处理

#去掉时间
# ratings = ratings.drop('timestamp', axis=1)

# songs['tags'] = songs.apply(lambda row : row['tags'].split(",")[0],axis=1)


# #去掉title列
# movies.drop(['title'],axis=1,inplace=True)

rating_song = pd.merge(ratings,songs,how='left',on="song")
# rating_song = pd.merge(left=ratings,
#          right=songs,
#          left_on='song',
#          right_on='song')

# replace gender values with 0,1
# users['gender'].replace({'F':0,'M':1},inplace=True)

# #replace age with an ordered list for the age
# users['age'].replace({1:0,18:1, 25:2, 35:3, 45:4, 50:5, 56:6 },inplace=True)

# #one got encode the zipcode column
# #users = pd.get_dummies(users,prefix=['zipcode'],columns=["zipCode"],drop_first=True)
# users.drop(['zipCode'],axis=1,inplace=True)

#%%

final_df = pd.merge(rating_song,users,how='left',on='user')

final_df.head()

'''

def encoder(df, cols=None):
    if cols == None:
        cols = list(df.select_dtypes(include=['object']).columns)

    val_types = dict()
    for c in cols:
        val_types[c] = df[c].unique()

    val_to_idx = dict()
    for k, v in val_types.items():
        val_to_idx[k] = {o: i for i, o in enumerate(val_types[k])}

    for k, v in val_to_idx.items():
        df[k] = df[k].apply(lambda x: v[x])

    return val_to_idx, df


#用于将数据分解为测试集和训练集并嵌入数据的函数
def data_processing(df, wide_cols, embeddings_cols, continuous_cols, target,
                    scale=False, def_dim=8):


    if type(embeddings_cols[0]) is tuple:
        emb_dim = dict(embeddings_cols)
        embeddings_cols = [emb[0] for emb in embeddings_cols]
    else:
        emb_dim = {e:def_dim for e in embeddings_cols}
    deep_cols = embeddings_cols+continuous_cols

    # Extract the target and copy the dataframe so we don't mutate it
    # internally.
    Y = np.array(df[target])
    all_columns = list(set(wide_cols + deep_cols ))
    df_tmp = df.copy()[all_columns]


    # 提取可在以后进行热编码的分类列名
    categorical_columns = list(df_tmp.select_dtypes(include=['object']).columns)


    encoding_dict,df_tmp = encoder(df_tmp)
    encoding_dict = {k:encoding_dict[k] for k in encoding_dict if k in deep_cols}
    embeddings_input = []
    for k,v in encoding_dict.items():
        embeddings_input.append((k, len(v), emb_dim[k]))

    df_deep = df_tmp[deep_cols]
    deep_column_idx = {k:v for v,k in enumerate(df_deep.columns)}


    if scale:
        scaler = StandardScaler()
        for cc in continuous_cols:
            df_deep[cc]  = scaler.fit_transform(df_deep[cc].values.reshape(-1,1))

    df_wide = df_tmp[wide_cols]
    del(df_tmp)
    dummy_cols = [c for c in wide_cols if c in categorical_columns]
    df_wide = pd.get_dummies(df_wide, columns=dummy_cols)

    X_train_deep, X_test_deep = train_test_split(df_deep.values, test_size=0.3, random_state=1463)
    X_train_wide, X_test_wide = train_test_split(df_wide.values, test_size=0.3, random_state=1463)
    y_train, y_test = train_test_split(Y, test_size=0.3, random_state=1981)

    group_dataset = dict()
    train_dataset = namedtuple('train_dataset', 'wide, deep, labels')
    test_dataset  = namedtuple('test_dataset' , 'wide, deep, labels')
    group_dataset['train_dataset'] = train_dataset(X_train_wide, X_train_deep, y_train)
    group_dataset['test_dataset']  = test_dataset(X_test_wide, X_test_deep, y_test)
    group_dataset['embeddings_input']  = embeddings_input
    group_dataset['deep_column_idx'] = deep_column_idx
    group_dataset['encoding_dict'] = encoding_dict

    return group_dataset


#数据设置
# wide_cols = ['song_year','tags','user','song']
# embeddings_cols = [('tags',20), ('user',100), ('song',100)]
wide_cols = ['song_year','user','song']
embeddings_cols = [('user',100), ('song',100)]
crossed_cols = ()
continuous_cols = ["song_year"]
target = 'rating'

#拆分数据并生成嵌入
data_processed = data_processing(
    final_df, wide_cols,
    embeddings_cols,
    continuous_cols,
    target,
    scale=True)


use_cuda = torch.cuda.is_available()

#加载数据集
class DatasetLoader(Dataset):
    def __init__(self, data):

        self.X_wide = data.wide
        self.X_deep = data.deep
        self.Y = data.labels

    def __getitem__(self, idx):

        xw = self.X_wide[idx]
        xd = self.X_deep[idx]
        y  = self.Y[idx]

        return xw, xd, y

    def __len__(self):
        return len(self.Y)



#类定义广度和深度神经网络
class NeuralNet(nn.Module):

    def __init__(self,
                 wide_dim,
                 embeddings_input,
                 continuous_cols,
                 deep_column_idx,
                 hidden_layers,
                 dropout,
                 encoding_dict,
                 n_class):

        super(NeuralNet, self).__init__()
        self.wide_dim = wide_dim
        self.deep_column_idx = deep_column_idx
        self.embeddings_input = embeddings_input
        self.continuous_cols = continuous_cols
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.encoding_dict = encoding_dict
        self.n_class = n_class
        self.loss_values=[]

        # 创建要穿过深侧的嵌入层
        for col,val,dim in self.embeddings_input:
            setattr(self, 'emb_layer_'+col, nn.Embedding(val, dim))

        # 如果指定，则使用下拉框构建深侧隐藏层
        input_emb_dim = np.sum([emb[2] for emb in self.embeddings_input])
        # input_emb_dim为float64
        # input_emb_dim = int(input_emb_dim)
        self.linear_1 = nn.Linear(input_emb_dim+len(continuous_cols), self.hidden_layers[0])
        if self.dropout:
            self.linear_1_drop = nn.Dropout(self.dropout[0])
        for i,h in enumerate(self.hidden_layers[1:],1):
            setattr(self, 'linear_'+str(i+1), nn.Linear( self.hidden_layers[i-1], self.hidden_layers[i] ))
            if self.dropout:
                setattr(self, 'linear_'+str(i+1)+'_drop', nn.Dropout(self.dropout[i]))

        # 将模型的wide侧和deep侧连接到输出神经元
        self.output = nn.Linear(self.hidden_layers[-1]+self.wide_dim, self.n_class)


    def compile(self, optimizer="Adam", learning_rate=0.001, momentum=0.0):

        self.activation, self.criterion = None, F.mse_loss

        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        if optimizer == "RMSprop":
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)

        self.method = 'regression'


    def forward(self, X_w, X_d):

        # Deep 侧
        emb = [getattr(self, 'emb_layer_'+col)(X_d[:,self.deep_column_idx[col]].long())
               for col,_,_ in self.embeddings_input]
        if self.continuous_cols:
            cont_idx = [self.deep_column_idx[col] for col in self.continuous_cols]
            cont = [X_d[:, cont_idx].float()]
            deep_inp = torch.cat(emb+cont, 1)
        else:
            deep_inp = torch.cat(emb, 1)

        x_deep = F.relu(self.linear_1(deep_inp))
        if self.dropout:
            x_deep = self.linear_1_drop(x_deep)
        for i in range(1,len(self.hidden_layers)):
            x_deep = F.relu( getattr(self, 'linear_'+str(i+1))(x_deep) )
            if self.dropout:
                x_deep = getattr(self, 'linear_'+str(i+1)+'_drop')(x_deep)

        # Deep + Wide 侧
        wide_deep_input = torch.cat([x_deep, X_w.float()], 1)

        if not self.activation:
            out = self.output(wide_deep_input)
        else:
            out = self.activation(self.output(wide_deep_input))

        return out


    def fit(self, dataset, n_epochs, batch_size):

        widedeep_dataset = DatasetLoader(dataset)
        train_loader = torch.utils.data.DataLoader(dataset=widedeep_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        # 将模型设置为评估模式，以便不应用退出
        net = self.train()
        for epoch in range(n_epochs):
            total=0
            correct=0
            for i, (X_wide, X_deep, target) in enumerate(train_loader):
                X_w = Variable(X_wide)
                X_d = Variable(X_deep)
                y = (Variable(target).float() if self.method != 'multiclass' else Variable(target))

                if use_cuda:
                    X_w, X_d, y = X_w.cuda(), X_d.cuda(), y.cuda()

                self.optimizer.zero_grad()
                y_pred =  net(X_w, X_d)
                y_pred = torch.squeeze(y_pred)
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()

                if self.method != "regression":
                    total+= y.size(0)
                    if self.method == 'logistic':
                        y_pred_cat = (y_pred > 0.5).squeeze(1).float()
                    if self.method == "multiclass":
                        _, y_pred_cat = torch.max(y_pred, 1)
                    correct+= float((y_pred_cat == y).sum().data[0])
            self.loss_values.append(loss.item())
            print ('Epoch {} of {}, Loss: {}'.format(epoch+1, n_epochs,
                                                     round(loss.item(),3)))


    def predict(self, dataset):


        X_w = Variable(torch.from_numpy(dataset.wide)).float()
        X_d = Variable(torch.from_numpy(dataset.deep))

        if use_cuda:
            X_w, X_d = X_w.cuda(), X_d.cuda()

        # set the model in evaluation mode so dropout is not applied
        net = self.eval()
        pred = net(X_w,X_d).cpu()
        if self.method == "regression":
            return pred.squeeze(1).data.numpy()
        if self.method == "logistic":
            return (pred > 0.5).squeeze(1).data.numpy()
        if self.method == "multiclass":
            _, pred_cat = torch.max(pred, 1)
            return pred_cat.data.numpy()




    def get_embeddings(self, col_name):
        params = list(self.named_parameters())
        emb_layers = [p for p in params if 'emb_layer' in p[0]]
        emb_layer  = [layer for layer in emb_layers if col_name in layer[0]][0]
        embeddings = emb_layer[1].cpu().data.numpy()
        col_label_encoding = self.encoding_dict[col_name]
        inv_dict = {v:k for k,v in col_label_encoding.items()}
        embeddings_dict = {}
        for idx,value in inv_dict.items():
            embeddings_dict[value] = embeddings[idx]

        return embeddings_dict


# 网络建立
wide_dim = data_processed['train_dataset'].wide.shape[1]
n_unique = len(np.unique(data_processed['train_dataset'].labels))
n_class = 1

deep_column_idx = data_processed['deep_column_idx']
embeddings_input= data_processed['embeddings_input']
encoding_dict   = data_processed['encoding_dict']
hidden_layers = [100,50]
dropout = [0.5,0.2]


use_cuda = torch.cuda.is_available()

model = NeuralNet(
    wide_dim,
    embeddings_input,
    continuous_cols,
    deep_column_idx,
    hidden_layers,
    dropout,
    encoding_dict,
    n_class)
model.compile(optimizer='Adam')
if use_cuda:
    model = model.cuda()

#训练模型
train_dataset = data_processed['train_dataset']
model.fit(dataset=train_dataset, n_epochs=5, batch_size=60)

#%%

test_dataset  = data_processed['test_dataset']

#%%

y_pred = model.predict(dataset=test_dataset)


y = test_dataset.labels

#获取测试MSE
mean_squared_error(y_pred,y)

#获取测试MAE
mean_absolute_error(y_pred,y)


#可视化模型的训练误差
plt.plot(model.loss_values)
plt.ylabel('Train Error')
plt.xlabel('epoch')
plt.title("Train error for models initial settings")
plt.show()


# In[]
## 超参数调谐

### 对于梯度下降算法

models = []
methods = ["Adam","RMSprop"]

for method in methods:
    model = NeuralNet(
        wide_dim,
        embeddings_input,
        continuous_cols,
        deep_column_idx,
        hidden_layers,
        dropout,
        encoding_dict,
        n_class)
    model.compile(optimizer=method)
    if use_cuda:
        model = model.cuda()
    model.fit(dataset=train_dataset, n_epochs=5, batch_size=60)
    models.append(model)


for model in models:
    plt.plot(np.arange(5),model.loss_values)
plt.ylabel('Train Error')
plt.xlabel('epoch')

plt.legend(methods, loc='upper left')
plt.title("Train error for different GD algorithms")
plt.show()


for model in models:
    print("for "+ str(model.optimizer))
    y_pred = model.predict(dataset=test_dataset)
    y = test_dataset.labels
    print("test mean squared error: "+str(mean_squared_error(y_pred,y)))
    print("test mean absolute error: "+ str(mean_absolute_error(y_pred,y)))


### 优化 dropout values
droupouts = [0,0.1,0.5]
models_dropout=[]

for droupout in droupouts:
    dropout = [droupout,droupout]
    model = NeuralNet(
        wide_dim,
        embeddings_input,
        continuous_cols,
        deep_column_idx,
        hidden_layers,
        dropout,
        encoding_dict,
        n_class)
    model.compile(optimizer="Adam")
    if use_cuda:
        model = model.cuda()
    model.fit(dataset=train_dataset, n_epochs=3, batch_size=60)
    models_dropout.append(model)


for model in models_dropout:
    plt.plot(np.arange(3),model.loss_values)
plt.ylabel('Train Error')
plt.xlabel('epoch')

plt.legend(droupouts, loc='upper left')
plt.title("Train error for different dropouts")
plt.show()


for model in models_dropout:
    print("for drououts: "+ str(model.dropout))
    y_pred = model.predict(dataset=test_dataset)
    y = test_dataset.labels
    print("test mean squared error: "+str(mean_squared_error(y_pred,y)))
    print("test mean absolute error: "+ str(mean_absolute_error(y_pred,y)))


## 用最优超参数评估模型
# Dropout 对于神经网络单元，按照一定的概率将其暂时从网络中丢弃
dropout = [0.5,0.5]
model = NeuralNet(
    wide_dim,
    embeddings_input,
    continuous_cols,
    deep_column_idx,
    hidden_layers,
    dropout,
    encoding_dict,
    n_class)
model.compile(optimizer="Adam")
if use_cuda:
    model = model.cuda()
model.fit(dataset=train_dataset, n_epochs=10, batch_size=60)


plt.plot(np.arange(10),model.loss_values)
plt.ylabel('Train Error')
plt.xlabel('epoch')

plt.title("Train error for optimal model")
plt.show()


y_pred = model.predict(dataset=test_dataset)
y = test_dataset.labels
print("test mean squared error: "+str(mean_squared_error(y_pred,y)))
print("test mean absolute error: "+ str(mean_absolute_error(y_pred,y)))


### 计算不同子集运行模型的时间
unique_users = final_df['user'].unique()
user_quantiles = np.arange(0.1,1,0.1)
runtimes = []


import time
for quantile in user_quantiles:
    start_time = time.time()
    subset_users = unique_users[:int(len(unique_users)*quantile)]
    subset_df = final_df.loc[final_df['user'].isin(subset_users)]
    data_processed = data_processing(
        subset_df, wide_cols,
        embeddings_cols,
        continuous_cols,
        target,
        scale=True)
    model = NeuralNet(
        wide_dim,
        embeddings_input,
        continuous_cols,
        deep_column_idx,
        hidden_layers,
        dropout,
        encoding_dict,
        n_class)
    model.compile(optimizer='Adam')
    if use_cuda:
        model = model.cuda()
    train_dataset = data_processed['train_dataset']
    model.fit(dataset=train_dataset, n_epochs=1, batch_size=60)
    end_time = time.time()
    total_time = end_time- start_time
    print("total time:" + str(total_time))
    runtimes.append(total_time)


#%%

plt.plot(user_quantiles,runtimes)
plt.ylabel('Runtime')
plt.xlabel('quantile size')

plt.title("Runtime for optimal model for different subset size")
plt.show()
