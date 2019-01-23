import os
import pickle
import numpy as np
from sklearn.externals import joblib
from utils import str2wv, word_embedding_dim
import config as cfg

from gensim.models import KeyedVectors

word_vector_model = KeyedVectors.load_word2vec_format(cfg.BIO_NLP_VEC, binary=True)

type = 'test'  # 'train'  'test'   'debug'
if type == 'test':
    crf = joblib.load('./model/with_BI_crf_model.m')
path = type.join(['dataset/', '/', '_with_pos_tag.data'])
pkl_path = type.join(['dataset/', '/', f'_with_BI_crf_max_cls_len{cfg.MAX_CLS_LEN}.pkl'])


def infos2features(words, pos_tags):
    '''将一个样本的信息转换成特征
        input:
            words:  样本句子分词之后的单词序列，  list
            pos_tags:   对应的单词词性序列，  list

        output:
            features: 对应的单词特征序列，    list
                    单个单词特征feature的序列为dict，窗口大小为3，包括大小写、数字、字母、词性等特征'''
    words_num = len(words)

    features = []
    for i in range(words_num):
        feature = {
            'word.lower()': words[i].lower(),
            'word[-3:]': words[i][-3:],
            'word[-2:]': words[i][-2:],
            'word[-1:]': words[i][-1:],  #
            'word.isupper()': words[i].isupper(),
            'word.istitle()': words[i].istitle(),
            'word.isdigit()': words[i].isdigit(),
            'word.isalnum()': words[i].isalnum(),  #
            'word.islower()': words[i].islower(),  #
            'postag': pos_tags[i],
        }
        if i > 0:
            feature.update({
                '-1:word.lower()': words[i - 1].lower(),
                '-1:word[-3:]': words[i - 1][-3:],  #
                '-1:word[-2:]': words[i - 1][-2:],  #
                '-1:word[-1:]': words[i - 1][-1:],  #
                '-1:word.istitle()': words[i - 1].istitle(),
                '-1:word.isupper()': words[i - 1].isupper(),
                '-1:word.isdigit()': words[i - 1].isdigit(),  #
                '-1:word.isalnum()': words[i - 1].isalnum(),  #
                '-1:word.islower()': words[i - 1].islower(),  #
                '-1:postag': pos_tags[i - 1],
            })
        else:
            feature['BOS'] = True  # begin of sentence

        if i < words_num - 1:
            feature.update({
                '+1:word.lower()': words[i + 1].lower(),
                '+1:word[-3:]': words[i + 1][-3:],  #
                '+1:word[-2:]': words[i + 1][-2:],  #
                '+1:word[-1:]': words[i + 1][-1:],  #
                '+1:word.istitle()': words[i + 1].istitle(),
                '+1:word.isupper()': words[i + 1].isupper(),
                '+1:word.isdigit()': words[i + 1].isdigit(),  #
                '+1:word.isalnum()': words[i + 1].isalnum(),  #
                '+1:word.islower()': words[i + 1].islower(),  #
                '+1:postag': pos_tags[i + 1],
            })
        else:
            feature['EOS'] = True  # end of sentence
        features.append(feature)

    return features


try:
    text = open(path).read()
except:
    text = open(path, encoding="utf-8").read()

all_data = []
all_info = text.strip('\n').split('\n\n')
for info in all_info:
    # 每个样本的处理
    infos = info.strip('\n').split('\n')
    words = infos[0].split()  # 单词序列, list
    pos_tags = infos[1].split()  # 词性序列, list
    one_data = {}
    # 转化为词向量, 200*200
    # str2wv重写过, word_vector_model只在create_data_set.py中加载到内存， 而不是在每次import utils * 之后都加载一次
    one_data["sentence"] = str2wv(words, word_vector_model).reshape(1, cfg.SENTENCE_LENGTH,  # 1 is the num of sentence.
                                                                    word_embedding_dim)
    one_data["str"] = infos[0]  # 文本内容, str
    words_num = len(words)
    labels = infos[2].split("|")  # 标注
    one_data["ground_truth_bbox"] = []
    one_data["ground_truth_cls"] = []
    one_data["region_proposal"] = []

    # GT范围: train时不做处理，全部为GT; test时为crf模型预测结果
    gt_position = crf.predict([infos2features(words, pos_tags)])[0] if type == 'test' else ['GT' for i in
                                                                                            range(words_num)]
    # 生成gt_bbox和gt_cls
    for item in labels:
        rl, cls = item.split(' G#')
        rl = rl.split(',')
        tuple_ = int(rl[0]), int(rl[1]) - 1
        one_data["ground_truth_bbox"].append(tuple_)
        one_data["ground_truth_cls"].append(cfg.LABEL.index(cls) + 1)

    # 根据GT范围生成候选框，
    for start_i in range(0, words_num):
        region_domain = min(cfg.MAX_CLS_LEN + 1, words_num - start_i + 1)  # 不超过entity最大长度和样本长度
        for length_j in range(1, region_domain):
            # 如果滑动窗口内存在BG，则从候选框集中移除
            if gt_position[start_i].find('BG') != -1 or gt_position[start_i + length_j - 1].find('BG') != -1:
                break
            tuple_ = (start_i, start_i + length_j - 1)  # 左闭右闭
            one_data["region_proposal"].append(tuple_)
    all_data.append(one_data)

with open(pkl_path, "wb") as f:
    pickle.dump(all_data, f)
