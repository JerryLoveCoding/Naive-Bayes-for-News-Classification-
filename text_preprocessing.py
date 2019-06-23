import os
import importlib
import NB
import random
import jieba
from sklearn.naive_bayes import MultinomialNB
def TextProcessing(folder_path, test_size = 0.2):
    """
    TextProcessing用于预处理数据，将其切分后，分成训练数据和测试数据，并建立词汇表
    :param folder_path:
    :param test_size:
    :return: all_words_list词汇表（不重复，按词频降序排列），train_data_list训练数据，test_data_list测试数据，train_class_list训练标签，test_class_list测试标签
    """
    folder_list = os.listdir(folder_path)
    data_list = []  # 数据集的数据
    class_list = []  # 数据集的标签

    for folder in folder_list:
        new_folder_path = os.path.join(folder_path,folder)
        files = os.listdir(new_folder_path)
        j = 1
        #遍历每一个txt文件
        for file in files:
            if j >100:
                break
            with open(os.path.join(new_folder_path, file), 'r', encoding='UTF-8') as f:
                raw = f.read()
            word_cut = jieba.cut(raw)  # jieba库的精简模式，返回一个可迭代的generator
            word_list = list(word_cut)  # 将generator形的转化为list形的
            data_list.append(word_list)  # 将当前这条数据加入data_list
            class_list.append(folder)  # 将当前这条数据的文件夹加入到class_list
            j += 1
    data_class_list = list(zip(data_list, class_list))  # 将数据的list和标签的list对应压缩成元组字典
    random.shuffle(data_class_list)  # 将data_class_list打乱顺序
    index = int(len(data_class_list)*test_size)+1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list, train_class_list = zip(*train_list)  # 对训练数据进行解压
    test_data_list, test_class_list = zip(*test_list)  # 对测试数据进行解压
    all_words_dict = {}  # 统计训练集词频(出现次数)
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict.keys():  # keys()以列表返回一个字典所有的键(键值对的键)。
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1
    # 根据键的值倒序排序，dict.items()是以列表返回可遍历（键-值）元组数组
    # sorted() 函数对所有可迭代的对象进行排序操作。
    # key主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
    # reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)#对词频字典(元组)的第二位（f[1]）进行比较
    all_words_list, all_words_nums = zip(*all_words_tuple_list)  # 解压缩
    all_words_list = list(all_words_list)  # 转换成列表
    # all_words_list词汇表（不重复,按词频降序排列），train_data_list训练数据，test_data_list测试数据，train_class_list训练标签，test_class_list测试标签
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


def MakeWordsSet(words_file):
    """
    处理文件每行的首尾指定字符
    :param words_file: 文件路径
    :return: return一个去掉首尾空格换行符的
    """
    words_set = set()  # set() 函数创建一个无序不重复元素集，也称集合，可进行关系测试，删除重复数据，还可以计算交集、差集、并集
    with open(words_file, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            word = line.strip()  # strip方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
            if len(word) > 0:
                words_set.add(word)
    return words_set

def word_dict(all_words_list,deleteN, stopwords_set=set()):
    """
    word_dict函数是用deleteN和停止词处理all_words_list词汇表
    :param all_words_list:词汇表（不重复，按照词频降序排列）
    :param deleteN:需要删除的前N个最高词频的词汇，因为根据打印的all_words_list,前N个最高词频的词往往都是停止词或者是废话词
    :param stopwords_set:停止词集合
    :return:feature_words,特征词集
    """
    feature_words = []
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000:  # 如果特征词汇的维度是1000维
            break
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1<len(all_words_list[t])<5:  # isdigit()方法检测字符串是否只由数字组成。
            feature_words.append(all_words_list[t])
        n += 1
    return feature_words

def TextFeatures(train_data_list, test_data_list, feature_words):
    """
    对测试数据集和训练数据集进行特征处理，如果特征词集的词汇在数据集中出现了就置1，否则置0
    :param train_data_list: 训练数据集
    :param test_data_list: 测试数据集
    :param feature_words: 特征词集
    :return:
    """
    def text_features(text, feature_words):
        #出现在特征词集就置1
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features

    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list

def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    MNB = MultinomialNB()  # MultinomialNB是先验概率符合多项式分布的朴素贝叶斯
    classifier = MNB.fit(train_feature_list, train_class_list)
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy





def makefeature():
    """
    与NB文件（分类单个目标新闻的文件）进行交互
    :return:特征集，训练数据集，训练数据标签集
    """
    folder_path = "./SogouC/Sample"
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path)
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)  # 生成stopwords_set的集合
    feature_words = word_dict(all_words_list, 150, stopwords_set)
    return feature_words, train_data_list, train_class_list


if __name__ == '__main__':
    folder_path = "./SogouC/Sample"
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path)
    # 停止词stopword，比如介词冠词连词等，我们称它为停止词。比如在、也、的、它、为这些词都是停止词
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)  # 生成stopwords_set的集合
    test_accuracy_list = []
    feature_words = word_dict(all_words_list, 150, stopwords_set)
    train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
    test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
    test_accuracy_list.append(test_accuracy)
    ave = lambda c: sum(c) / len(c)
    print(ave(test_accuracy_list))




