from model import *
from data_parser import *
from utils import *
from get_word2vec import *

with open('./config/output_file.yaml', 'rb') as f:
    params = yaml.load(f)


if __name__ == '__main__':

    for file_in in [params['file_content_bikesharing'], params['file_content_xiongan'],
                    params['file_content_gaotie'], params['file_content_beima']]:
        content_parse(
            file_in=file_in,
            file_out='./output/news_tensite_xml_withoutEnter.dat') 

    get_wv(
        file_in='./output/news_tensite_xml_withoutEnter.dat',
        file_out=params['file_word2vec_bin']
    )

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

    data_parser_gaotie = Data_Parser(
        file_stopwords=params['file_stopwords'],
        file_word2vec_bin=params['file_word2vec_bin'],
        file_sent2vec=params['file_sent2vec_gaotie'],
        file_doc2vec=params['file_doc2vec_gaotie']
    )

    data_parser_beima = Data_Parser(
        file_stopwords=params['file_stopwords'],
        file_word2vec_bin=params['file_word2vec_bin'],
        file_sent2vec=params['file_sent2vec_beima'],
        file_doc2vec=params['file_doc2vec_beima']
    )
   
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

    data_parser_gaotie.get_sent_vec(
        file_in=params['file_sent_gaotie'],
        num_keywords=5
    )

    data_parser_gaotie.get_doc_vec(
        file_in=params['file_doc_gaotie'],
        num_keywords=10
    )

    data_parser_beima.get_sent_vec(
        file_in=params['file_sent_beima'],
        num_keywords=5
    )

    data_parser_beima.get_doc_vec(
        file_in=params['file_doc_beima'],
        num_keywords=10
    )

    dir_name = {
        '共享单车': 'bikesharing',
        '雄安': 'xiongan',
        '北马': 'beima',
        '高铁': 'gaotie'
    }

    for k, v in dir_name.items():
        for method in ['ap', 'kmeans']:
            keywords_cluster(
                file_in=params['file_word_{}'.format(v)],
                file_out='./output/temp/{}/cluster_table_word_using_{}.csv'.format(
                    k, method),
                method=method,
                n_cluster=20,
                show_or_write='write'
            )

            sent_or_doc_cluster(
                file_in=[params['file_sent_{}'.format(v)], params['file_word_{}'.format(
                    v)], params['file_sent2vec_{}'.format(v)]],
                file_out=['./output/temp/{}/cluster_table_sent_using_{}.csv'.format(k, method),
                          './output/temp/{}/cluster_digest_sent_using_{}.csv'.format(k, method)],
                feature='vec',
                method=method,
                n_cluster=20,
                show_or_write='write'
            )

            sent_or_doc_cluster(
                file_in=[params['file_doc_{}'.format(v)], params['file_word_{}'.format(
                    v)], params['file_doc2vec_{}'.format(v)]],
                file_out=['./output/temp/{}/cluster_table_doc_using_{}.csv'.format(k, method),
                          './output/temp/{}/cluster_digest_doc_using_{}.csv'.format(k, method)],
                feature='vec',
                method=method,
                n_cluster=20,
                show_or_write='write'
            )
        pass

