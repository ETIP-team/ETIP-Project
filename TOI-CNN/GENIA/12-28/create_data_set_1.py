# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2018-10-07

import os
import pickle

from utils import *
import config as cfg

from gensim.models import KeyedVectors

word_vector_model = KeyedVectors.load_word2vec_format(cfg.BIO_NLP_VEC, binary=True)

path = "dataset/train/train.data"
pkl_path = f"dataset/train/train_base_max_cls_len{cfg.MAX_CLS_LEN}.pkl"
try:
    text = open(path).read()
except:
    text = open(path, encoding="utf-8").read()

ls_data = text.strip('\n').split("\n\n")
for index in range(len(ls_data)):
    ls_data[index] = [item.strip(" ").strip("|") for item in ls_data[index].strip('\n').split("\n")]

all_data = []
for sub_ls in ls_data:
    if len(sub_ls) > 2:
        print(sub_ls)
        sub_ls = sub_ls[len(sub_ls) - 2:]
    if len(sub_ls) < 1:
        continue
    one_data = {}
    one_data["sentence"] = str2wv(sub_ls[0].split(" "), word_vector_model).reshape(1, cfg.SENTENCE_LENGTH,
                                                                                   word_embedding_dim)  # make sentence sample.
    one_data["str"] = sub_ls[0]
    sentence_length = len(sub_ls[0].split(" "))

    labels = sub_ls[1].split("|")

    one_data["ground_truth_bbox"] = []
    one_data["ground_truth_cls"] = []
    one_data["region_proposal"] = []
    for item in labels:
        tuple_ = tuple([int(item) for item in item.split(" ")[0].split(",")])
        # 11-16 right boundary -1
        tuple_ = tuple_[0], tuple_[1] - 1
        one_data["ground_truth_bbox"].append(tuple_)
        # make it a number.
        one_data["ground_truth_cls"].append(
            cfg.LABEL.index(item[item.find("#") + 1:]) + 1)  # index 0   is the background.
    if len(one_data["ground_truth_cls"]) != len(one_data["ground_truth_bbox"]):
        print(sub_ls)
        print("Label wrong")
        wait = True
    for start_i in range(0, sentence_length):
        for length_j in range(1, min(cfg.MAX_CLS_LEN + 1, sentence_length - start_i + 1)):
            tuple_ = (start_i, start_i + length_j - 1)
            one_data["region_proposal"].append(tuple_)
    all_data.append(one_data)
with open(pkl_path, "wb") as f:
    pickle.dump(all_data, f)
