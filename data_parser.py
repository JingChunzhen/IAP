import word2vec
import re
import jieba
import pickle
import hashlib
import yaml
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from utils import load_stopwords, line_parse


class Data_Parser(object):
    """
    data preprocessing
    to get word vectors, sent vectors, doc vectors, one-hot vectors etc
    to test the cluster effect 
    """

    def __init__(self, file_stopwords, file_word2vec_bin, file_sent2vec, file_doc2vec):
        self.stop_words = load_stopwords(file_stopwords)
        self.file_word2vec_bin = file_word2vec_bin
        self.file_sent2vec = file_sent2vec
        self.file_doc2vec = file_doc2vec
        pass

    def _get_hash(self, line):
        sha1 = hashlib.sha1(line).hexdigest()
        return sha1

    def content_parse(self, file_in, file_out, rid_stopwords):
        re_string = r'[^\u4e00-\u9fa5a-zA-Z0-9]'
        regx = re.compile(re_string)
        with open(file_in, 'rb') as f_in, open(file_out, 'a+', encoding='utf-8') as f_out:
            for line in f_in:
                new_line_string = line_parse(
                    line, regx, rid_stopwords, self.stop_words)
                f_out.write(new_line_string)

    def get_word_vec(self, file_in, size):
        """
        Args:
            file_in (string): 
            szie (int): size of word embeddings
        the model stored in self.file_word2vec_bin
        """
        word2vec.word2vec(file_in, self.file_word2vec_bin, size, verbose=False)

    def get_sent_vec(self, file_in, num_keywords):
        """        
        Args:
            file_in (string)
            num_keywords (int): num of keywords selected from a weight dict sorted by tf-idf 
        Returns:
            dict: sentence (string) -> index (int) hash sentence to sha-1 
            dict: index (int) -> vector (float array)
            dict: index (int) -> one hot vector (int array)
        """
        sent_id = dict()
        id_sentvec = dict()
        id_sentonehot = dict()
        # with open(self.file_word2vec, 'rb') as f_in:
        #     word_vec = pickle.load(f_in)
        model = word2vec.load(self.file_word2vec_bin)

        re_string = r'[^\u4e00-\u9fa5a-zA-Z0-9]'
        regx = re.compile(re_string)

        corpus = []
        with open(file_in, 'rb') as f_in:
            i = 0
            for line in f_in:
                sha1 = self._get_hash(line)
                sent_id[sha1] = i
                line = line_parse(line, regx, True, self.stop_words)
                corpus.append(line)
                i += 1

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
            temp_keys = [k for k, v in temp]

            sentvec = np.zeros(shape=100)
            c = 0
            for index in temp_keys:
                if tfidf_weight[i][index] != 0:
                    try:
                        sentvec += model[words[index]]  # TODO
                        c += 1
                    except:
                        pass

            if c != 0:
                # RuntimeWarning: invalid value encountered in true_divide
                id_sentvec[i] = sentvec / c
            else:  # except for zero division error
                id_sentvec[i] = sentvec

            id_sentonehot[i] = onehot_weight[i]

        with open(self.file_sent2vec, 'wb') as f_out:
            pickle.dump(sent_id, f_out)
            pickle.dump(id_sentvec, f_out)
            pickle.dump(id_sentonehot, f_out)

    def get_doc_vec(self, file_in, num_keywords):
        """        
        Args:
            file_in (string)
            num_keywords (int): num of keywords selected from a weight dict sorted by tf-idf 
        Returns:
            dict: doc (string) -> index (int) hash doc to sha-1 
            dict: index (int) -> vector (float array)
            dict: index (int) -> one hot vector (int array) 
                can PCA handle this sort of high dimensional data ?  
        """
        doc_id = dict()
        id_docvec = dict()
        id_doconehot = dict()

        re_string = r'[^\u4e00-\u9fa5a-zA-Z0-9]'
        regx = re.compile(re_string)

        # with open(self.file_word2vec, 'rb') as f_in:
        #     word_vec = pickle.load(f_in)

        model = word2vec.load(self.file_word2vec_bin)

        corpus = []
        with open(file_in, 'rb') as f_in:
            i = 0
            for line in f_in:
                sha1 = self._get_hash(line)
                doc_id[sha1] = i
                line = line_parse(line, regx, True, self.stop_words)
                corpus.append(line)
                i += 1

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
            temp_keys = [k for k, v in temp]

            docvec = np.zeros(shape=100)
            c = 0
            for index in temp_keys:
                if tfidf_weight[i][index] != 0:
                    try:
                        c += 1
                        docvec += model[words[index]]  # TODO
                    except:
                        pass

            if c != 0:
                id_docvec[i] = docvec / c
            else:
                id_docvec[i] = docvec

            id_doconehot[i] = onehot_weight[i]

        with open(self.file_doc2vec, 'wb') as f_out:
            pickle.dump(doc_id, f_out)
            pickle.dump(id_docvec, f_out)
            pickle.dump(id_doconehot, f_out)
        pass

    pass  # end of class


if __name__ == "__main__":
    file_config = './config/output_file.yaml'

    with open(file_config, 'rb') as f:
        params = yaml.load(f)

    data_parser_bikesharing = Data_Parser(
        file_stopwords=params['file_stopwords'],
        file_word2vec_bin=params['file_word2vec_bin'],
        file_sent2vec=params['file_sent2vec_bikesharing'],
        file_doc2vec=params['file_doc2vec_bikesharing']
    )

    data_parser_xiongan = Data_Parser(
        file_stopwords=params['file_stopwords'],
        file_word2vec_bin=params['file_word2vec_bin'],
        file_sent2vec=params['file_sent2vec_xiongan'],
        file_doc2vec=params['file_doc2vec_xiongan']
    )
    '''
    data_parser.content_parse(
        file_in=params['file_content'],
        file_out=params['file_normalized_content'],
        rid_stopwords=False
    )
    data_parser.get_word_vec(
        file_in=params['file_normalized_content'],
        size=50
    )
    '''
    data_parser_bikesharing.get_sent_vec(
        file_in=params['file_sent_bikesharing'],
        num_keywords=5
    )

    data_parser_bikesharing.get_doc_vec(
        file_in=params['file_doc_bikesharing'],
        num_keywords=10
    )

    data_parser_xiongan.get_sent_vec(
        file_in=params['file_sent_xiongan'],
        num_keywords=5
    )

    data_parser_xiongan.get_doc_vec(
        file_in=params['file_doc_xiongan'],
        num_keywords=10
    )
