# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2019-04-04


import numpy as np
from attention_neww2vmodel import geniaDataset

from config import Config


def word2ids(words: list, word_dict: geniaDataset) -> list:
    """:param
        words: list of str
        word_dict: Dataset word dict
       :returns
        ids: list of id for each word(int) """
    ls_id = []
    for word in words:
        try:
            ls_id.append(word_dict.vocabToNew[word])
        except KeyError:
            ls_id.append(0)
    return ls_id


def data_prepare(config: Config, path: str, word_dict: geniaDataset) -> (list, list, list):
    # read file path and ...
    #
    """    :param config: Config file
    :param path: data file path
    :param word_dict: used to prepare word id.
    :return: data: list of sub list withs info: word ids
        cause the length could be different so we need to save it in list with different shapes
        [[1,2,3], [2,3],[3,4]]
    :return: label: list of sub list which is ["nested_level=0"[0,1,2], "nested_level" [0,0,1]]
     info in "" not included.
    Attention: for each seq the depth could be different."""

    data_list = open(path, "r", encoding="gbk").read().strip("\n").split("\n\n")

    data = []
    data_str = []
    labels = []
    for data_index in range(len(data_list)):

        temp = data_list[data_index].split("\n")

        max_nested_level = config.max_nested_level
        one_seq_labels = [[] for i in range(max_nested_level)]

        sentence_info = [item.split(" ") for item in temp]

        words = [sentence_info[i][0] for i in range(len(sentence_info))]

        word_ids = word2ids(words, word_dict)

        for nested_index in range(max_nested_level):
            one_layer_labels = [config.bio_labels.index(sentence_info[i][nested_index + 1]) for i in
                                range(len(sentence_info))]

            one_seq_labels[nested_index].extend(one_layer_labels)

        empty_label = [0 for i in range(len(sentence_info))]
        empty_label_count = one_seq_labels.count(empty_label)

        [one_seq_labels.remove(empty_label) for i in range(empty_label_count)]  # remove the empty labels.

        data.append(word_ids)
        data_str.append(words)
        labels.append(one_seq_labels)

    return data, data_str, labels


def output_level(write_file, level, id2label, index, mode):
    write_file.write(mode + f" level {level + 1}: " + "\t".join([id2label[i] for i in index]) + '\n')


def output_summary(write_file, words, id2label, predict_candidates, gt_entities):
    write_file.write("gt:  ")
    for item in gt_entities:
        write_file.write(id2label[item[2]] + ' [' + ' '.join(words[item[0]: item[1]]) + '] ')
    write_file.write("\npre:\n")
    for i, pre in enumerate(predict_candidates):
        hit = "T" if pre in gt_entities else "F"
        s, e, l = predict_candidates[i]
        write_file.write(hit + f"\t{id2label[l]}\t{' '.join(words[s: e])} [{s},{e}]\n")
    write_file.write("______________________________\n\n")


def output_sent(write_file, words):
    write_file.write(" ".join(words) + "\n")


def find_entities_relax(config: Config, predict_bio_label_index: list) -> list:
    predict_entities = []
    labels = []
    for label_index in predict_bio_label_index:
        if label_index > 0:
            labels.append(config.labels[int((label_index - 1) / 2)])
        else:
            labels.append("O")

    start_index = -1
    current_label = "O"

    for label_index in range(len(labels)):
        if labels[label_index] == "O":
            if start_index > -1:
                predict_entities.append((start_index, label_index, config.labels.index(labels[start_index])))
                start_index = -1
                current_label = "O"
        else:
            if current_label == "O":  # previous is bg.
                start_index = label_index
            else:  # previous is gt
                if current_label != labels[label_index]:
                    predict_entities.append((start_index, label_index, config.labels.index(labels[start_index])))
                    start_index = label_index
                    current_label = labels[label_index]

        if label_index == len(labels) - 1:  # if the final, end started entity.
            if start_index > -1:
                predict_entities.append((start_index, len(labels), config.labels.index(labels[start_index])))

    return predict_entities


def find_entities(config: Config, predict_bio_label_index: list) -> list:
    """

    :param config: used bio labels, labels in here.
    :param predict_bio_label_index: list of int, which is the bio label index
    :return:predict_entities: list of tuples, which format is (start_index, end_index, label)
    """
    predict_entities = []
    str_labels = [config.bio_labels[label_index] for label_index in predict_bio_label_index]

    start_index = end_index = -1

    for str_label_index in range(len(str_labels)):
        if str_labels[str_label_index] == "O":
            if start_index > -1:
                end_index = str_label_index
                predict_entities.append((start_index, end_index, config.labels.index(str_labels[start_index][2:])))
                start_index = end_index = -1

        if str_labels[str_label_index].startswith("B-"):
            if start_index > -1:
                end_index = str_label_index
                predict_entities.append((start_index, end_index, config.labels.index(str_labels[start_index][2:])))
                start_index = end_index = str_label_index  # new begin
            else:
                start_index = end_index = str_label_index  # start count this entity.

        if str_labels[str_label_index].startswith("I-"):
            if start_index > -1:  # only with a B- start already!
                if str_labels[start_index][2:] == str_labels[str_label_index][2:]:
                    end_index += 1  # continue.
        if str_label_index == len(str_labels) - 1:  # if the final, end started entity.
            if start_index > -1:
                end_index = len(str_labels)
                predict_entities.append((start_index, end_index, config.labels.index(str_labels[start_index][2:])))

    return predict_entities
