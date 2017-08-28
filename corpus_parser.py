import gensim
from gensim.models import Word2Vec
import re
import jieba


def _line_parse(line):  
    ret_string = None
    line = line.strip()
    line = line.decode('gb18030')
    if line == "":
        pass
    else:
        re_string = r'<content>(.*)</content>'
        regx = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9]')
        line = re.findall(re_string, line)  # return a list
        if len(line) != 0:
            line = line[0]
            line = regx.sub("", line.strip())
            line_seg = jieba.cut(line)
            ret_string = " ".join(line_seg)
    return ret_string


def corpus_parse(file_in, file_out):
    with open(file_in, 'rb') as f_in, open(file_out, 'w+') as f_out:
        for line in f_in:
            line = _line_parse(line)
            if line is not None:
                print(line)
                if line != "":
                    f_out.write(line)
                    f_out.write('\n')


def get_word2vec(file_in, file_out):
    """以string形式写入一个文件中，读取时得到的是一个二维的list 并作为参数传入word2vec中去
    """
    file_model = './output/wordvec_test.bin'
    with open('./output/news_tensite_xml.smarty-normalized.dat', 'rb') as f:
        c = 0
        corpus = []
        for line in f:
            line = line.decode('utf-8')
            if line != "":
                corpus.append(line.split(' '))        
        model = Word2Vec(corpus, size=100, window=5, min_count=5, workers=4)
        model.save(file_model)
    pass


def test_this_model():
    file_model = './output/wordvec_test.bin'
    model = Word2Vec.load(file_model)
    print(model.wv[u"中国"])
    a = model.wv.similarity(u'中国', u'法国')
    b = model.wv.similarity('中国', '对于')
    print(a)
    print(b)
    pass


if __name__ == '__main__':
    # test_this_model()
    # get_word2vec()
    # test_this_model()
    # corpus_parse()
    # with open('./output/news_tensite_xml.smarty-normalized.dat', 'rb') as f:
    #     c = 0
    #     corpus = []
    #     for line in f:
    #         line = line.decode('utf-8')
    #         if line != "":
    #             corpus.append(line)

    # get_word2vec()

    # print(corpus)
    #test_this_model()

    # corpus_parse()
    # get_word2vec()

    file_in = './data/news_tensite_xml.dat'
    file_out = './output/news_tensite_xml-normalized.dat'
    corpus_parse(file_in, file_out)
    #corpus_parse()
    pass
