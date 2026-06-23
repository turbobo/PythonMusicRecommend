import numpy as np
import networkx as nx
import gensim

# 随机游走生成顶点序列
def randomWalk(_g, _corpus_num, _deep_num, _current_word):
    _corpus = []
    for i in range(_corpus_num):
        sentence = [_current_word]
        current_word = _current_word
        count = 0
        while count<_deep_num:
            count+=1
            _node_list = []
            _weight_list = []
            for _nbr, _data in _g[current_word].items():
                _node_list.append(_nbr)
                _weight_list.append(_data['weight'])
            _ps = [float(_weight) / sum(_weight_list) for _weight in _weight_list]
            sel_node = roulette(_node_list, _ps)
            sentence.append(sel_node)
            current_word = sel_node
        _corpus.append(sentence)
    return _corpus

def roulette(_datas, _ps):
    return np.random.choice(_datas, p=_ps)


# 生成有向图网络
G = nx.DiGraph()
path = './graph.txt'
word_list = []
with open(path,'r') as f:
    for line in f:
        cols = line.strip().split(',')
        G.add_weighted_edges_from([(cols[0], cols[1], float(cols[2]))])
        word_list.append(cols[0])
        G.add_weighted_edges_from([(cols[1], cols[0], float(cols[2]))])
        word_list.append(cols[1])

word_set = set(word_list)

num = 10				# 每个节点开始生成num条顶点序列
deep_num = 20			 #每个序列深度为deep_num

sentence_file = open('./GraphSentence.txt','w')
k = 1
for word in word_set:
    print(k)
    k+=1
    corpus = randomWalk(G, num, deep_num, word)
    # print(corpus)
    for cols in corpus:
        sentences = '\t'.join(cols)
        sentence_file.write(sentences+'\n')


# word2vector
with open('./GraphSentence.txt','r') as f:
    sentences = []
    for line in f:
        cols = line.strip().split('\t')
        sentences.append(cols)



model = gensim.models.Word2Vec(sentences, sg=1, size=300, alpha=0.025, window=3, min_count=1, max_vocab_size=None, sample=1e-3, seed=1, workers=45, min_alpha=0.0001, hs=0, negative=20, cbow_mean=1, hashfxn=hash, iter=5, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=1e4)


outfile = './test'
fname = './testmodel'
# save
model.save(fname)
model.wv.save_word2vec_format(outfile + '.model.bin', binary=True)
model.wv.save_word2vec_format(outfile + '.model.txt', binary=False)


fname = './testmodel'
model = gensim.models.Word2Vec.load(fname)
nearest10 = model.most_similar('子')
print(nearest10 )