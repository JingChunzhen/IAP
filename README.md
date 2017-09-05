# IAP
该项目使用爬虫得到的语料进行聚类
语料中包含关键词，关键句，关键段
## 关键词的聚类
该部分首先使用Word2vec训练词向量，之后使用训练得到的词向量进行聚类，由于给的语料较少，故使用[搜狗实验室的语料](https://www.sogou.com/labs/resource/list_yuliao.php)来进行训练，算法使用ap和kmeans，使用新的语料和ap算法得到的效果很好
其中改进了sklearn中ap算法的部分，该部分在cluster_algos.py文件中展示，将向量之间的求距离由欧式距离改为了余弦距离
## 句的聚类
该部分首先将关键句中的停用词去掉，之后使用tf-idf筛选出关键词，筛选出的关键词数可指定，将筛选出的关键词词向量相加之后求均值作为该关键句的句向量
聚类时，使用句向量来进行，取得了不错的效果
同时也使用one-hot向量来表示每个句子，并使用one-hot向量来进行聚类，但效果不佳
## 段落的聚类

