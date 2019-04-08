# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2019-04-04


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


def data_prepare(config: Config, path: str, word_dict: geniaDataset) -> (list, list):
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
        labels.append(one_seq_labels)

    return data, labels
