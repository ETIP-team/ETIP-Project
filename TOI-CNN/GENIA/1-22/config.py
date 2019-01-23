# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2018-10-03

K_FOLD = 1
LABEL = ['DNA', 'RNA', 'cell_type', 'protein', 'cell_line', 'bg']

CLASSES_NUM = len(LABEL) - 1

TRAIN_FLAG = False
SW_TECH_FLAG = True

WORD_SEGMENTATION_METHOD = "genia"  # "nlpir"

MAX_CLS_LEN = 32
SENTENCE_LENGTH = 200
NN_SAVE_PATH = "model/genia/"
ALL_WRITE_PATH = "result/genia/"

# if WORD_SEGMENTATION_METHOD == "jieba":
#     SENTENCE_LENGTH = 72  # 67
#     NN_SAVE_PATH = "model/CNN_jieba/"
#     ALL_WRITE_PATH = "result/CNN_jieba_sw/" if SW_TECH_FLAG else "result/CNN_jieba/"
# elif WORD_SEGMENTATION_METHOD == "nlpir":
#     SENTENCE_LENGTH = 83
#     NN_SAVE_PATH = "model/CNN_nlpir/"
#     ALL_WRITE_PATH = "result/CNN_nlpir_sw/" if SW_TECH_FLAG else"result/CNN_nlpir/"
# else:
#     NN_SAVE_PATH = "model/CNN_other/"
#     SENTENCE_LENGTH = 100
#     ALL_WRITE_PATH = "result/CNN_other_sw/" if SW_TECH_FLAG else "result/CNN_other/"

# Model  Setting
WORD_EMBEDDING_DIM = 200
LEARNING_RATE = 1e-4
KERNAL_LENGTH = 5
L2_BETA = 0.001
FEATURE_MAPS_NUM = 36

POOLING_OUT = 2  # to be declared.

# Training Setting
TH_IOU_TRAIN = 0.6
TH_IOU_NMS = 0.05
TH_IOU_TEST = 0.6
TH_SCORE = 0.6

# File Path
# JIEBA_WV_MODEL = "./model/word_vector_model/all.seg300w50428.model"
# JIEBA_USER_DICT = "./model/word_vector_model/jieba_userdict.txt"
# NLPIR_WV_MODEL = "./model/word_vector_model/nlpir.model"

BIO_NLP_VEC = "./model/word_vector_model/bio_nlp_vec.tar/bio_nlp_vec/PubMed-shuffle-win-2.bin"
# BIO_NLP_VEC = "./model/word_vector_model/wikipedia-pubmed-and-PMC-w2v.bin"
TEST_PATH = "result/CNN_genia_sw/"
