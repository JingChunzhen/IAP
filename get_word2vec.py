import word2vec
import re
import jieba
import yaml

with open('./config/output_file.yaml', 'rb') as f:
    params = yaml.load(f)


def corpus_parse(file_in, file_out):
    '''
    处理的是搜狗实验室的语料
    '''
    re_string = r'<content>(.*)</content>'
    regx = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9]')
    with open(file_in, 'rb') as f_in, open(file_out, 'a+') as f_out:
        for line in f_in:
            line = line.strip()
            line = line.decode('gb18030')
            if line == "":
                pass
            else:
                line = re.findall(re_string, line)
                if len(line) != 0:
                    line = line[0]
                    line = regx.sub("", line.strip())
                    line_seg = jieba.cut(line)
                    ret_string = " ".join(line_seg)
                    print(ret_string)
                    f_out.write(ret_string)
                    f_out.write(' ')


def _line_parser(line):
    re_string = r'[^\u4e00-\u9fa5a-zA-Z0-9]'
    regx = re.compile(re_string)
    line = line.strip()
    ret_string = None
    if line == '':
        pass
    else:
        line = regx.sub('', line.decode('utf-8'))
        line_seg = jieba.cut(line)
        ret_string = ' '.join(line_seg)
    return ret_string


def content_parse(file_in, file_out):
    '''
    处理的是聚类的语料
    '''
    with open(file_in, 'rb') as f_in, open(file_out, 'a+', encoding='utf-8') as f_out:
        for line in f_in:
            new_line_string = _line_parser(line)
            f_out.write(new_line_string)
            f_out.write(' ')


def get_wv(file_in, file_out):
    word2vec.word2vec(file_in, file_out, size=100, verbose=False)
    print("OK")
    pass


def test_this_model():
    file_model = './output/word2vec_test.bin'
    model = word2vec.load(file_model)
    print(model['北马'].shape)
    indexes, metrics = model.cosine('北马', n=100)
    print(model.vocab[indexes])
    pass


    
