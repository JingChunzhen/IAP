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


class Data_Parser(object):
    """
    data preprocessing
    to get word vectors, sent vectors, doc vectors, one-hot vectors etc
    to test the cluster effect 
    """

    def __init__(self, file_stopwords, file_word2vec_bin, file_sent2vec, file_doc2vec):
        self.stop_words = self._load_stop_words(file_stopwords)
        self.file_word2vec_bin = file_word2vec_bin        
        self.file_sent2vec = file_sent2vec
        self.file_doc2vec = file_doc2vec
        pass

    def _load_stop_words(self, file_stopwords):
        stop_words = []
        with open(file_stopwords, 'rb') as f_sw:
            for line in f_sw:
                line = line.decode('utf-8').strip()
                stop_words.append(line)
        return stop_words

    def _line_parser(self, line, regx, rid_stopwords):
        """preprocess the data 
        Args:
            line (string): data needed to be preprocessed 
            rid_stopwords (boolean): True for rid of stop words False for else 
        Return:
            string
        """        
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
                    if word not in self.stop_words:
                        word_list.append(word)
                ret_string = ' '.join(word_list)
            else:
                ret_string = ' '.join(line_seg)
        return ret_string

    def _get_hash(self, line):
        sha1 = hashlib.sha1(line).hexdigest()
        return sha1

    def content_parse(self, file_in, file_out, rid_stopwords):
        re_string = r'[^\u4e00-\u9fa5a-zA-Z0-9]'
        regx = re.compile(re_string)
        with open(file_in, 'rb') as f_in, open(file_out, 'a+', encoding='utf-8') as f_out:
            for line in f_in:
                new_line_string = self._line_parser(line, regx, rid_stopwords)
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
                line = self._line_parser(line, regx, True)
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
                        sentvec += model[words[index]] # TODO
                        c += 1 
                    except:
                        pass

            if c != 0:
                id_sentvec[i] = sentvec / c  # RuntimeWarning: invalid value encountered in true_divide
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
                line = self._line_parser(line, regx, True)
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
    
    data_parser = Data_Parser(
        file_stopwords=params['file_stopwords'],
        file_word2vec_bin=params['file_word2vec_bin'],        
        file_sent2vec=params['file_sent2vec_bikesharing'],
        file_doc2vec=params['file_doc2vec_bikesharing']
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
    data_parser.get_sent_vec(
        file_in=params['file_sent_bikesharing'],
        num_keywords=5
    )

    data_parser.get_doc_vec(
        file_in=params['file_doc_bikesharing'],
        num_keywords=10
    )
