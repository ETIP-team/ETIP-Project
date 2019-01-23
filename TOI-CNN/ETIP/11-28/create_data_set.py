# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2018-10-07

import os
import pickle
import jieba
from utils import *
import config as cfg

# from gensim.models import Word2Vec
# jieba.load_userdict("model/jieba_userdict.txt")
# Word2Vec.load("model/word_vector_model/all.seg300w50428.model")

# path = "dataset/train/train_relabeled_data_10_30/"
# pkl_path = "dataset/train/train_relabeled_data_pkl_11_16_rb_modify/"
path = "dataset/test/test_relabeled_data_10_30/"
pkl_path = "dataset/test/test_relabeled_data_pkl_11_16_rb_modify/"

# mention_hyper_graph
# for file in os.listdir(path):
#     text = open(path + file).read() if path.find("test") > -1 else open(path + file, encoding="utf-8").read()
#     text = text.replace("\n\n\n", "\n\n")
#     ls_data = text.split("\n\n")
#     for index in range(len(ls_data)):
#         ls_data[index] = ls_data[index].split("\n")
#     all_data = []
#     for sub_ls in ls_data:
#         if len(sub_ls) < 3:
#             continue
#         one_data = {}
#         one_data["sentence"] = str2wv(sub_ls[0].split(" ")).reshape(1, cfg.SENTENCE_LENGTH, word_embedding_dim)
#         one_data["str"] = sub_ls[0]
#         # print(one_data["sentence"].shape)
#         # exit()
#         sentence_length = len(sub_ls[0].split(" "))
#         labels = sub_ls[2].split("|")
#         one_data["ground_truth_bbox"] = []
#         one_data["ground_truth_cls"] = []
#         one_data["region_proposal"] = []
#         # ls_ground_truth = []
#         for item in labels:
#             tuple_ = tuple([int(item) for item in item.split(" ")[0].split(",")])
#             # ls_ground_truth.append(tuple_)
#             one_data["ground_truth_bbox"].append(tuple_)
#             # make it a number.
#             one_data["ground_truth_cls"].append(cfg.LABEL.index(item[item.find("#")+1:])+1)
#         for start_i in range(0, sentence_length):
#             for length_j in range(1, sentence_length-start_i+1):
#                 tuple_ = (start_i, start_i+length_j)
#                 # if tuple_ in one_data["ground_truth_bbox"]:
#                 #     continue
#                 one_data["region_proposal"].append(tuple_)
#         all_data.append(one_data)
#     with open(pkl_path+file[:-5]+".pkl", "wb") as f:
#         pickle.dump(all_data, f)

# self label
for file in os.listdir(path):
    try:
        text = open(path + file).read()
    except:
        text = open(path + file, encoding="utf-8").read()
    # text = text.replace("\n\n\n", "\n\n")
    ls_data = text.split("\n\n")
    for index in range(len(ls_data)):
        ls_data[index] = ls_data[index].split("\n")
    all_data = []
    for sub_ls in ls_data:
        if len(sub_ls) > 2:
            print(sub_ls)
            sub_ls = sub_ls[len(sub_ls) - 2:]
        if len(sub_ls) < 1:
            continue
        one_data = {}
        one_data["sentence"] = str2wv(sub_ls[0].split(" ")).reshape(1, cfg.SENTENCE_LENGTH, word_embedding_dim)
        one_data["str"] = sub_ls[0]
        # print(one_data["sentence"].shape)
        # exit()
        sentence_length = len(sub_ls[0].split(" "))
        try:
            labels = sub_ls[1].split("|")
        except:
            wait = True
        one_data["ground_truth_bbox"] = []
        one_data["ground_truth_cls"] = []
        one_data["region_proposal"] = []
        # ls_ground_truth = []
        for item in labels:
            try:
                tuple_ = tuple([int(item) for item in item.split(" ")[0].split(",")])
                # 11-16 right boundary -1
                tuple_ = tuple_[0], tuple_[1] - 1
            except:
                wait = True
            # ls_ground_truth.append(tuple_)
            one_data["ground_truth_bbox"].append(tuple_)
            # make it a number.
            try:
                one_data["ground_truth_cls"].append(cfg.LABEL.index(item[item.find("#") + 1:]) + 1)
            except:
                wait = True
        for start_i in range(0, sentence_length):
            for length_j in range(1, sentence_length - start_i + 1):
                tuple_ = (start_i, start_i + length_j - 1)
                # if tuple_ in one_data["ground_truth_bbox"]:
                #     continue
                one_data["region_proposal"].append(tuple_)
        all_data.append(one_data)
    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)
    with open(pkl_path + file[:-5] + ".pkl", "wb") as f:
        pickle.dump(all_data, f)
