import matplotlib.pyplot as plt
import numpy as np
# 第三方库
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from surprise import KNNBasic
from surprise import Reader, Dataset, accuracy
from surprise import SVD
from surprise.model_selection import KFold

# In[]:

data = pd.read_csv('data/metadata/lastfm_unique_tags.txt',
                   sep='\t', header=None, names=['tag', 'count'])


# 用户的歌曲播放总量的分布
# 字典user_playcounts记录每个用户的播放总量
tag_counts = {}
for tag, group in data.groupby('tag'):
    tag_counts[tag] = group['count'].sum()


# In[]:


# 作图
# sns.displot(list(tag_counts.values()), bins=5000, kde=False)
sns.displot(list(tag_counts.values()), kde=False)
plt.xlim(25, 200)
plt.xlabel('count')
plt.ylabel('nums of tag')
plt.show()


temp_tag = [tag for tag in tag_counts.keys() if tag_counts[tag] > 200]
temp_tagcounts = [counts for tag, counts in tag_counts.items() if counts > 200]

print('标签大于100数量占总体标签数量的比例为', str(round(len(temp_tag)/len(tag_counts), 4)*100)+'%')
print('标签大于100产生的播放总量占总体播放总量的比例为', str(round(sum(temp_tagcounts) / sum(tag_counts.values())*100, 4))+'%')
print('标签大于100产生的数据占总体数据的比例为', str(round(len(data[data.tag.isin(temp_tag)])/len(data)*100, 4))+"%")


# 标签大于100数量占总体标签数量的比例为 1.79%
# 标签大于100产生的播放总量占总体播放总量的比例为 67.6858%
# 标签大于100产生的数据占总体数据的比例为 1.7924%

# 标签大于200数量占总体标签数量的比例为 0.86%    4480个标签   共 522366个标签
# 标签大于200产生的播放总量占总体播放总量的比例为 59.7875%
# 标签大于200产生的数据占总体数据的比例为 0.8574%
