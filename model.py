
import word2vec
import yaml
import numpy as np
from cluster_algos import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances

file_word2vec_bin = './output/word2vec_test.bin'


def keywords_cluster(file_in, method):
    with open(file_in, 'rb') as f_in:
        content = f_in.read()
        content = content.decode('utf-8')
        words = content.split(' ')
    model = word2vec.load(file_word2vec_bin)
    new_words = []
    x = []
    for word in words:
        try:
            x.append(model[word])
            new_words.append(word)
        except:
            pass

    X = np.array(x)
   
    if method.lower() == 'ap':
        af = AffinityPropagation(affinity='cosine').fit(X)
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        n_clusters_ = len(cluster_centers_indices)

        labels = labels.tolist()
        word_cluster = dict()
        cluster_words = dict()

        for i in range(len(labels)):
            word_cluster[new_words[i]] = labels[i]

        for w, c in word_cluster.items():
            if c not in cluster_words:
                cluster_words[c] = []
                cluster_words[c].append(w)
            else:
                cluster_words[c].append(w)

        for c, ws in cluster_words.items():
            print(c)
            print(ws)

        pass
    
    elif method.lower() == 'kmeans':
        
        pass
    
    else:

        pass

def keywords_cluster_using_KM(file_in):

    pass

if __name__ == '__main__':
    keywords_cluster_using_AP('./data/共享单车-语料/共享单车keywords.txt')
    pass