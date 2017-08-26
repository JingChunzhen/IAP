

import word2vec
import re
import jieba
import pickle
import hashlib
import yaml
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class Data_Parser(object):
    """
    data preprocessing
    to get word vectors, sent vectors, doc vectors, one-hot vectors etc
    to test the cluster effect 
    """

    def __init__(self, file_stopwords, file_word2vec_bin, file_word2vec, file_sent2vec, file_doc2vec):
        self.stop_words = self._load_stop_words(file_stopwords)
        self.file_word2vec_bin = file_word2vec_bin
        self.file_word2vec = file_word2vec
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

    def _line_parser(self, line, rid_stopwords):
        """preprocess the data 
        Args:
            line (string): data needed to be preprocessed 
            rid_stopwords (boolean): True for rid of stop words False for else 
        Return:
            string
        """
        re_string = r'[^\u4e00-\u9fa5a-zA-Z0-9]'
        regx = re.compile(re_string)
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
        with open(file_in, 'rb') as f_in, open(file_out, 'a+', encoding='utf-8') as f_out:
            for line in f_in:
                new_line_string = self._line_parser(line, rid_stopwords)
                f_out.write(new_line_string)

    def get_word_vec(self, file_in, size):
        """
        Args:
            file_in (string): 
            szie (int): size of word embeddings
        Return:
            dict: word (string) -> vector (float array)

        self.file_word2vec_bin a word2vec model stored 
        self.file_word2vec a dict stored: key ->  string value -> word2vec shape = (size,)
        the content in self.file_word2vec is the final result of this function
        """
        word2vec.word2vec(file_in, self.file_word2vec_bin, size, verbose=False)
        model = word2vec.load(self.file_word2vec_bin)
        word_length = model.vocab.shape[0]

        word_vec = dict()
        for index in range(word_length):
            word_vec[model.vocab[index]] = model.vectors[index]

        with open(self.file_word2vec, 'wb') as f_out:
            pickle.dump(word_vec, f_out)

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
        with open(self.file_word2vec, 'rb') as f_in:
            word_vec = pickle.load(f_in)

        corpus = []
        with open(file_in, 'rb') as f_in:
            i = 0
            for line in f_in:
                sha1 = self._get_hash(line)
                sent_id[sha1] = i
                line = self._line_parser(line, True)
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

            sentvec = 0
            c = 0
            for index in temp_keys:
                if tfidf_weight[i][index] != 0:
                    if words[index] in word_vec:
                        c += 1                    
                        sentvec += word_vec[words[index]]  # TODO 
                    else:
                        pass                    

            try:
                id_sentvec[i] = sentvec / c
            except: # except for zero division error
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
        with open(self.file_word2vec, 'rb') as f_in:
            word_vec = pickle.load(f_in)

        corpus = []
        with open(file_in, 'rb') as f_in:
            i = 0
            for line in f_in:
                sha1 = self._get_hash(line)
                doc_id[sha1] = i
                line = self._line_parser(line, True)
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

            docvec = 0
            c = 0
            for index in temp_keys:
                if tfidf_weight[i][index] != 0:
                    if words[index] in word_vec:                        
                        c += 1 
                        docvec += word_vec[words[index]]  # TODO
                    else:
                        pass                    
            
            try:
                id_docvec[i] = docvec / c
            except:
                id_docvec[i] = docvec

            id_doconehot[i] = onehot_weight[i]

        with open(self.file_doc2vec, 'wb') as f_out:
            pickle.dump(doc_id, f_out)
            pickle.dump(id_docvec, f_out)
            pickle.dump(id_doconehot, f_out)
        pass

    pass  # end of class


if __name__ == "__main__":
    # should be stored into profiles
    file_stopwords = './data/stopwords.txt'
    file_word2vec_bin = './output/temp/共享单车_word2vec_bin.bin'
    file_word2vec = './output/temp/共享单车_word2vec.txt'
    file_sent2vec = './output/temp/共享单车_sent2vec.txt'
    file_doc2vec = './output/temp/共享单车_doc2vec.txt'

    file_content = './data/共享单车-语料/共享单车content.txt'
    file_normalized_content = './output/temp/共享单车_normalized_content.txt'
    file_sent = './data/共享单车-语料/共享单车keySentence.txt'
    file_doc = './data/共享单车-语料/共享单车KeyParagraphs.txt'

    data_parser = Data_Parser(
        file_stopwords=file_stopwords,
        file_word2vec_bin=file_word2vec_bin,
        file_word2vec=file_word2vec,
        file_sent2vec=file_sent2vec,
        file_doc2vec=file_doc2vec
    )

    data_parser.content_parse(
        file_in=file_content,
        file_out=file_normalized_content,
        rid_stopwords=False
    )

    data_parser.get_word_vec(
        file_in=file_normalized_content,
        size=50
    )

    data_parser.get_sent_vec(
        file_in=file_sent,
        num_keywords=5
    )

    data_parser.get_doc_vec(
        file_in=file_doc,
        num_keywords=10
    )
