# -*- coding:utf-8 -*-
#
# Created by ETIP-team
#
# On 2018-09-04


K_FOLD = 5
LABEL = ['保险期间', '合同终止', '责任免除',  '保障范围', '等待期', '给付条件', '给付金额']

CLASSES_NUM = len(LABEL)

TRAIN_FLAG = True
SW_TECH_FLAG = True



WORD_SEGMENTATION_METHOD = "jieba"  # "nlpir"

if WORD_SEGMENTATION_METHOD == "jieba":
    SENTENCE_LENGTH = 67
    NN_SAVE_PATH = "model/CNN_jieba/"
    ALL_WRITE_PATH = "result/CNN_jieba_sw/" if SW_TECH_FLAG else "result/CNN_jieba/"
elif WORD_SEGMENTATION_METHOD == "nlpir":
    SENTENCE_LENGTH = 83
    NN_SAVE_PATH = "model/CNN_nlpir/"
    ALL_WRITE_PATH = "result/CNN_nlpir_sw/" if SW_TECH_FLAG else"result/CNN_nlpir/"
else:
    NN_SAVE_PATH = "model/CNN_other/"
    SENTENCE_LENGTH = 100
    ALL_WRITE_PATH = "result/CNN_other_sw/" if SW_TECH_FLAG else "result/CNN_other/"


SEED_VALUE = 1024

# Model  Setting
max_cross_entropy = 0.20
min_cross_entropy = 0.08
LS_CROSS_ENTROPY_TH = [round(item/100, 2) for item in range(int(max_cross_entropy*100), int(min_cross_entropy*100)-1, -1)]
WORD_EMBEDDING_DIM = 300

KERNAL_LENGTH = 4
L2_BETA = 0.001
TH_C = 0.955
TH_S = 0.995
TH_IOU = 0.2

FEATURE_MAPS_NUM = 36
POOLING_HEIGHT = 4
POOLING_OUT = (SENTENCE_LENGTH-KERNAL_LENGTH + 1)//POOLING_HEIGHT

MIN_LENGTH_TH = 3
MIN_VOTER_NUM = 2

TH_P = 0.6
MIN_TH_P = 0.4

# File Path
JIEBA_WV_MODEL = "./model/word_vector_model/all.seg300w50428.model"
JIEBA_USER_DICT = "./model/word_vector_model/jieba_userdict.txt"
NLPIR_WV_MODEL = "./model/word_vector_model/nlpir.model"


TEST_PATH = "result/CNN_jieba_sw/"

