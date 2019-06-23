import os
import random
import jieba
import numpy as np
import text_preprocessing
from sklearn.naive_bayes import MultinomialNB

def DataProcessing(folder_path):
    """
    预处理要进行分类的那一个数据文件
    :param folder_path:
    :return:
    """
    data_list = []  # 数据集的数据
    class_list = []  # 数据集的标签
    with open(folder_path, 'r', encoding='utf-8') as f:
        raw = f.read()
    word_cut = jieba.cut(raw)  # jieba库的精简模式，返回一个可迭代的generator
    data_list = list(word_cut)  # 将generator形的转化为list形的

    word_list = {}  # 词频字典
    for word in data_list:
        if word in word_list.keys():
            word_list[word] += 1
        else:
            word_list[word] = 1
    words_tuple_list = sorted(word_list.items(), key=lambda f: f[1], reverse=True)  # 对词频字典降序排列
    words_list, words_nums = zip(*words_tuple_list)  # 解压缩
    words_list = list(words_list)
    return data_list, words_list, words_tuple_list
#处理训练数据
def dealtrain(data_list):
    """
    预处理训练数据集
    :param data_list:
    :return:
    """
    mylist = []  # 对训练数据进行按词频降序排列
    for row in data_list:
        word_list = {}  # 词频字典
        for word in row:
            if word in word_list.keys():
                word_list[word] += 1
            else:
                word_list[word] = 1
        words_tuple_list = sorted(word_list.items(), key=lambda f: f[1], reverse=True)  # 对词频字典降序排列
        words_list, words_nums = zip(*words_tuple_list)  # 解压缩
        words_list = list(words_list)
        mylist.append(words_list)
    return mylist



def wordtonumber(feature_words, data):
    """
    将预处理后的数据集（训练数据集或要分类的数据）转化为一个索引列表，每个元素和特征集的元素索引对应，如果不存在于特征集就置0
    :param feature_words:
    :param data:
    :return:
    """
    wtn = []  # 每个单词在feature_words中的索引
    for word in data:
        if word in feature_words:
            wtn.append(feature_words.index(word)+1)  # 加1的原因是如果不存在就置0，为了不和feature_words的首元素撞车所以索引值都+1
        else:
            wtn.append(0)  # 如果不在feature_words中，将其置为0（不置-1是因为MultinomialNB不允许出现负数）
    return wtn



def pre(simplify_data_index, simplify_train_data_list_index, train_class_list):
    """
    预测的函数,输出一个预测结果
    :param simplify_data_index:精简过后的要分类的目标新闻的索引列表
    :param simplify_train_data_list_index:精简过后的要分类的训练集的索引列表
    :param train_class_list:训练数据分类好的标签列表
    :return:
    """
    simplify_train_data_list = np.array(simplify_train_data_list_index)
    train_class_list = np.array(train_class_list)
    simplify_data_index = np.array(simplify_data_index).reshape(1, -1)
    MNB = MultinomialNB()  # MultinomialNB是先验概率符合多项式分布的朴素贝叶斯
    MNB.fit(simplify_train_data_list, train_class_list)
    pre1 = MNB.predict(simplify_data_index)  # 预测结果
    return pre1


def pretoword(pre1):
    """
    将预测结果变换成ClassList1表中对应的中文
    :param pre1:
    :return:
    """
    pre_dict = {}
    with open("./SogouC/ClassList1.txt", 'r', encoding='utf-8') as f:
        for raw in f:
            (key, value) = raw.strip().split(':')
            pre_dict[key] = value
    pre_word = pre_dict[pre1[0]]
    return pre_word


def sum(folder_path):
    """
    总函数，如果调用该文件就调用这个函数,输入文件路径（文件要是utf-8格式）
    输出一个预测结果
    :param folder_path:
    :return:
    """
    data_list, words_list, words_tuple_list = DataProcessing(folder_path)
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = text_preprocessing.MakeWordsSet(stopwords_file)  # 生成stopwords_set的集合
    simplify_data = text_preprocessing.word_dict(words_list, 5, stopwords_set)  # 精简的数据
    # 所有数据得到的特征集合
    feature_words, train_data_list, train_class_list = text_preprocessing.makefeature()
    train_class_list = list(train_class_list)
    train_data_list = dealtrain(train_data_list)

    # 精简训练集和训练数据
    simplify_train_data_list = []
    for i in train_data_list:
        simplify = text_preprocessing.word_dict(i, 5, stopwords_set)
        simplify_train_data_list.append(simplify[0:26])

    # 将精简数据化成特征集的索引
    simplify_data = simplify_data[0:26]
    simplify_data_index = wordtonumber(feature_words, simplify_data)
    simplify_train_data_list_index = []
    for data in simplify_train_data_list:
        simplify_train_data_list_index.append(wordtonumber(feature_words, data))
    print(simplify_data_index)
    print(simplify_train_data_list_index)

    pre1 = pre(simplify_data_index, simplify_train_data_list_index, train_class_list)



    return pre1

if __name__ == '__main__':
    folder_path = "./SogouC/Sample/C000008/16.txt"
    data_list, words_list, words_tuple_list = DataProcessing(folder_path)
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = text_preprocessing.MakeWordsSet(stopwords_file)  # 生成stopwords_set的集合
    simplify_data = text_preprocessing.word_dict(words_list, 5, stopwords_set)  # 精简的数据

    #所有数据得到的特征集合
    feature_words, train_data_list, train_class_list = text_preprocessing.makefeature()
    train_class_list = list(train_class_list)
    train_data_list = dealtrain(train_data_list)

    # 精简训练集和训练数据
    simplify_train_data_list = []
    for i in train_data_list:
        simplify = text_preprocessing.word_dict(i, 5, stopwords_set)
        simplify_train_data_list.append(simplify[0:26])

    # 将精简数据化成特征集的索引
    simplify_data = simplify_data[0:26]
    simplify_data_index = wordtonumber(feature_words, simplify_data)
    simplify_train_data_list_index = []
    for data in simplify_train_data_list:
        simplify_train_data_list_index.append(wordtonumber(feature_words, data))
    print(simplify_data_index)
    print(simplify_train_data_list_index)
    pre1 = pre(simplify_data_index, simplify_train_data_list_index, train_class_list)
    pre1_word = pretoword(pre1)  # 预测结果对应的中文
    print(pre1)
    print(pre1_word)


