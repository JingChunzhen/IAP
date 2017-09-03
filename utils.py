import re
import jieba
import pandas as pd
import linecache
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def load_stopwords(file_stopwords):
    stop_words = []
    with open(file_stopwords, 'rb') as f_sw:
        for line in f_sw:
            line = line.decode('utf-8').strip()
            stop_words.append(line)
    return stop_words


def line_parse(line, regx, rid_stopwords, stop_words):
    '''
    parse a line 
    Args:
        line (string): 
        regx : re.complier 
        rid_stopwords (bool): can either be True for rid or False for not rid 
        stop_words (list): stop words get from func load_stopwords stored here 
    Return:
        string: string of words seperated by space 
    '''
    line = line.strip()
    ret_string = None
    if line == '':
        pass
    else:
        line = regx.sub('', line.decode('utf-8'))
        line_seg = jieba.cut(line)
        if rid_stopwords:
            word_list = []
            for word in line_seg:
                if word not in stop_words:
                    word_list.append(word)
            ret_string = ' '.join(word_list)
        else:
            ret_string = ' '.join(line_seg)
    return ret_string


def show(original_file, cluster_ids):
    '''
    show the cluster result to the standard output 
    Args:
        original_file (string): original file of content (sentences or docs)
        cluster_ids (doc): key -> cluster ids (list)-> content ids 
    '''
    for cluster, ids in cluster_ids.items():
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print(cluster)
        for i in ids:
            line = linecache.getline(original_file, i)
            print(line)
        pass
    pass


def write_to(original_file, file_out, file_stopwords, id_cluster, num_keywords=5):
    '''
    visualize the clustering result stored in a file     
    Args:
        original_file (string): original corpus of sentences or docs 
        file_out (string): file name to store the table 
        id_cluster (dict): get from func sent_or_doc_cluster
        num_keywords (int): num of keywords of each content(sentence or doc)     
    '''

    re_string = r'[^\u4e00-\u9fa5a-zA-Z0-9]'
    regx = re.compile(re_string)

    stopwords = load_stopwords(file_stopwords)
    data = []
    original_corpus = []
    corpus = []

    with open(original_file, 'rb') as f:
        for original_line in f:
            original_corpus.append(original_line)
            line = line_parse(original_line, regx, True, stopwords)
            corpus.append(line)  # TODO a list or a string

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    onehot = vectorizer.fit_transform(corpus)
    tfidf = transformer.fit_transform(onehot)
    onehot_weight = onehot.toarray()
    tfidf_weight = tfidf.toarray()
    words = vectorizer.get_feature_names()

    for i in range(tfidf_weight.shape[0]):
        temp_dict = {}
        for j in range(tfidf_weight.shape[1]):
            temp_dict[j] = tfidf_weight[i][j]

        temp = sorted(temp_dict.items(), key=lambda d: d[1], reverse=True)[
            :num_keywords]
        keywords_list = [words[k] for k, v in temp]

        keywords_string = ' '.join(keywords_list)

        data.append([id_cluster[i], keywords_string,
                     original_corpus[i].decode('utf-8')])

    pd_dist = pd.DataFrame(data=data, index=None, columns=[
                           'cluster_id', 'key_words', 'content'])
    pd_dist = pd_dist.sort_values(by='cluster_id', ascending=True)
    pd_dist.to_csv(file_out)


def write_to_2(original_content_file, original_words_file, file_out, file_stopwords, id_cluster):
    '''
    '''
    data = []
    stopwords = load_stopwords(file_stopwords)
    re_string = r'[^\u4e00-\u9fa5a-zA-Z0-9]'
    regx = re.compile(re_string)

    with open(original_words_file, 'rb') as f_in:
        content = f_in.read().decode('utf-8')
        key_words = content.split(' ')

    with open(original_content_file, 'rb') as f:
        i = 0
        for original_line in f:
            #original_corpus.append(original_line)
            line = line_parse(original_line, regx, True, stopwords)

            words_list = line.split(' ')
            new_words = []
            for word in words_list:
                if word in key_words:
                    new_words.append(word)

            keywords_string = ' '.join(new_words)
            data.append([id_cluster[i], keywords_string, original_line.decode('utf-8')])

            i += 1

    pd_dist = pd.DataFrame(data=data, columns=[
                           'cluster_id', 'key_words', 'content'])
    pd_dist = pd_dist.sort_values(by='cluster_id', ascending=True)
    pd_dist.to_csv(file_out)
