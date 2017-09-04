import word2vec
import yaml
import pickle
import hashlib
import numpy as np
from cluster_algos import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from utils import load_stopwords, line_parse, show, \
    data_analysis_using_tfidf, data_analysis_using_keywords, data_analysis_using_keywords_2, \
    data_analysis_using_keywords_3


with open('./config/output_file.yaml', 'rb') as f:
    params = yaml.load(f)


def keywords_cluster(file_in, method, n_cluster):
    """keywords clustering
    Args:
        file_in (string): keywords
        method (string): cluster method can only be 'ap' or 'kmeans'
        n_cluster (int): cluster num for kmeans method 
    """
    with open(file_in, 'rb') as f_in:
        content = f_in.read()
        content = content.decode('utf-8')
        words = content.split(' ')
    model = word2vec.load(params['file_word2vec_bin'])
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
        word_cluster = {}
        cluster_words = {}

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
        km = KMeans(n_clusters=n_cluster).fit(X)
        labels = km.labels_.tolist()
        word_cluster = {}
        cluster_words = {}

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

    else:
        raise ValueError("Method must be 'ap' or "
                         "'kmeans'. Got %s instead"
                         % method)
        pass


def sent_or_doc_cluster(original_file, file_in, file_out, feature, method, n_cluster, show_or_write):
    """
    can be tested using one-hot, doc2vec, doc2vec self-made three features
    and can be tested using ap or kmeans 
    Args:
        original (string): file for sentences or docs 
        file_in (string): file preprocessed by data_parser with vectors and dict stored
        feature (string): can only be one-hot, vec (normalized vectors), doc2vec
        method (string): can either be 'ap' or 'kmeans'
        n_cluster (int): cluster num for kmeans method 
    """
    if feature.lower() == 'onehot':
        with open(file_in, 'rb') as f_in:
            content_id = pickle.load(f_in)
            id_vec = pickle.load(f_in)
            id_onehot = pickle.load(f_in)
            x = []
            for i, onehot in id_onehot.items():
                x.append(onehot.tolist())

            X = np.array(x)

            if method.lower() == 'ap':
                instance = AffinityPropagation(affinity='cosine').fit(X)
            elif method.lower() == 'kmeans':
                instance = KMeans(n_cluster=n_cluster).fit(X)

            labels = instance.labels_.tolist()
            id_cluster = {}
            cluster_ids = {}
            for i in range(len(labels)):
                id_cluster[i] = labels[i]

            for i, cluster in id_cluster.items():
                if cluster not in cluster_ids:
                    cluster_ids[cluster] = []
                    cluster_ids[cluster].append(i)
                else:
                    cluster_ids[cluster].append(i)
            pass
            if show_or_write == 'show':
                show(original_file, cluster_ids)
            else:
                data_analysis_using_keywords_2(
                    original_file=original_file,
                    file_out=file_out,
                    file_stopwords=params['file_stopwords'],
                    id_cluster=id_cluster,
                    num_keywords=20)
        pass

    elif feature.lower() == 'vec':
        with open(file_in, 'rb') as f_in:
            content_id = pickle.load(f_in)
            id_vec = pickle.load(f_in)
            id_onehot = pickle.load(f_in)
            x = []
            for i, vec in id_vec.items():
                x.append(vec.tolist())  # int object jas nor attribute 'tolist'

            X = np.array(x)

            if method.lower() == 'ap':
                instance = AffinityPropagation(affinity='cosine').fit(X)
            elif method.lower() == 'kmeans':
                instance = KMeans(n_clusters=n_cluster).fit(X)
            else:
                raise ValueError("Method must be 'ap' or "
                                 "'kmeans'. Got %s instead"
                                 % method)

            labels = instance.labels_.tolist()
            id_cluster = {}
            cluster_ids = {}
            for i in range(len(labels)):
                id_cluster[i] = labels[i]

            for i, cluster in id_cluster.items():
                if cluster not in cluster_ids:
                    cluster_ids[cluster] = []
                    cluster_ids[cluster].append(i)
                else:
                    cluster_ids[cluster].append(i)
            if show_or_write == 'show':
                show(original_file, cluster_ids)
            else:
                # write_to(
                #     original_file=original_file,
                #     file_out=file_out,
                #     file_stopwords=params['file_stopwords'],
                #     id_cluster=id_cluster,
                #     num_keywords=20)

                data_analysis_using_keywords_3(
                    file_in=original_file,
                    file_out=file_out,
                    original_words_file=params['file_word_bikesharing'],
                    file_stopwords=params['file_stopwords'],
                    cluster_ids=cluster_ids
                )
        pass
    elif feature.lower() == 'doc2vec':
        # word2vec.doc2vec
        pass
    else:
        raise ValueError(
            "Feature must be 'onehot' or 'vec' or 'doc2vec'. Got %s instead" % feature)
        pass

    pass


if __name__ == '__main__':

    #keywords_cluster(params['file_word_xiongan'], 'kmeans', 20)

    # sent_or_doc_cluster(
    #     original_file=params['file_doc_xiongan'],
    #     file_in=params['file_doc2vec_xiongan'],
    #     feature='vec',
    #     method='ap',
    #     n_cluster=10)
    # ValueError: setting an array element with a sequence

    sent_or_doc_cluster(
        original_file=params['file_sent_bikesharing'],
        file_in=params['file_sent2vec_bikesharing'],
        file_out='./output/temp/共享单车sent_cluster_data_analysis.csv',
        feature='vec',
        method='ap',
        n_cluster=20,
        show_or_write='write'
    )

    pass
