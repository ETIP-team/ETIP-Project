# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2019-04-03

import torch as t
from torch.autograd import Variable
import pandas as pd
import numpy as np

from bilstm_attention_crf import AttentionNestedNERModel
from crf_config import Config
from attention_neww2vmodel import geniaDataset

from utils import data_prepare, output_summary, output_sent, output_level
from utils import find_entities, find_entities_relax


def get_metrics(config: Config) -> pd.DataFrame:
    """

    :param config: Config file, used labels, metric_dicts
    :return: dataframe with metrics precision, recall, f1.
    """
    data_frame_data = []
    columns_name = ["Label", "Precision", "Recall", "F1"]
    all_tp = 0
    all_fp = 0
    all_fn = 0
    for label_index in range(len(config.labels)):
        one_row = []
        one_row.append(config.labels[label_index])
        true_positive = config.metric_dicts[label_index]["TP"]

        false_positive = config.metric_dicts[label_index]["FP"]
        false_negative = config.metric_dicts[label_index]["FN"]

        all_tp += true_positive
        all_fp += false_positive
        all_fn += false_negative

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
        f1_measure = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

        one_row.extend([precision, recall, f1_measure])

        data_frame_data.append(one_row)

    all_precision = all_tp / (all_tp + all_fp) if all_tp + all_fp > 0 else 0
    all_recall = all_tp / (all_tp + all_fn) if all_tp + all_fn > 0 else 0
    all_f1 = (2 * all_precision * all_recall) / (all_precision + all_recall) if all_precision + all_recall > 0 else 0

    data_frame_data.append(["Overall", all_precision, all_recall, all_f1])
    return pd.DataFrame(data_frame_data, columns=columns_name, index=None)


def evaluate(config: Config, predict_candidates: list, gt_entities: list):
    """

    :param config: Config file, used metric_dicts
    :param predict_candidates: list of tuples, which format (start_index, end_index, label)
    :param gt_entities: list of tuples, which format (start_index, end_index, label)
    :return:
    """
    gt_hit_list = [False] * len(gt_entities)

    for pred_entity in predict_candidates:
        pred_entity_label = pred_entity[2]
        if pred_entity in gt_entities:
            if gt_hit_list[gt_entities.index(pred_entity)] is False:
                config.metric_dicts[pred_entity_label]["TP"] += 1
                gt_hit_list[gt_entities.index(pred_entity)] = True
        else:
            config.metric_dicts[pred_entity_label]["FP"] += 1

    for gt_entity_index in range(len(gt_entities)):
        gt_entity_label = gt_entities[gt_entity_index][2]
        if gt_hit_list[gt_entity_index] is False:  # not hit yet
            config.metric_dicts[gt_entity_label]["FN"] += 1
    return


def process_one_nested_level_predict_result(config: Config, predict_bio_label_index: list,
                                            predict_candidates: list) -> (list, bool):
    """

    :param config: Config file, pass to find_entities function
    :param predict_bio_label_index: list of int, which is the bio label index
    :param predict_candidates: all meaningful and unduplicated candidates.
    :return predict_candidates: add entities in this level
    :return added_flag: boolean value to check if the candidates been updated or not.
    """
    # first process the predict labels:
    # predict_label_index = list((predict_bio_label_index - 1) / 2)  # notice that the label is the gt label.
    added_flag = False
    # for entity in find_entities(config, predict_bio_label_index):
    for entity in find_entities_relax(config, predict_bio_label_index):
        if entity not in predict_candidates:
            predict_candidates.append(entity)
            added_flag = True

    return predict_candidates, added_flag


def _test_one_sentence(config: Config, model: AttentionNestedNERModel, word_ids: list) -> list:  # todo
    """

    :param config: Config file, pass to process_one_nested_level_predict_result function.
    :param model: AttentionNestedNERModel
    :param word_ids: word_id for one sequence.
    :return: predict_candidates: predict entities in this sequence.
    """
    predict_candidates = []

    if config.cuda:
        one_seq_word_ids = Variable(t.Tensor([word_ids]).cuda().long())
    else:
        one_seq_word_ids = Variable(t.Tensor([word_ids]).long())

    predict_score, predict_result = model.predict(one_seq_word_ids)
    # todo process here.
    predict_result = np.array(predict_result)
    # predict_result = np.array(predict_result).reshape(config.max_nested_level, len(word_ids), 1,
    #                                                        len(config.bio_labels))
    # predict_result = predict_result.squeeze(2)  # remove batch num = 1
    # predict_result = predict_result.cpu().data.numpy()
    for nested_level in range(config.max_nested_level):
        predict_bio_label_index = predict_result[nested_level]
        # todo rectify.
        # predict_bio_label_index = rectify_bio_labels(predict_bio_label_index)
        # predict_bio_label_index = t.argmax(predict_result, dim=1).cpu().data.numpy()
        output_level(config.result_file, nested_level, config.bio_labels, predict_bio_label_index, "pre")

        predict_candidates, added_flag = process_one_nested_level_predict_result(config,
                                                                                 list(predict_bio_label_index),
                                                                                 predict_candidates)

    return predict_candidates


def _test(config: Config, model: AttentionNestedNERModel):
    """

    :param config: Config file pass to test_one_sentence, find_entities, evaluate functions.
    :param model: model after load parameters.
    :return:
    """
    # initialize the confusion matrix.
    config.metric_dicts = [{"TP": 0, "FP": 0, "FN": 0} for i in range(len(config.labels))]
    # start test.
    config.result_file = open(config.output_path, "w")
    for test_index in range(len(config.test_data)):
        output_sent(config.result_file, config.test_str[test_index])
        word_ids = config.test_data[test_index]
        gt_labels = config.test_label[test_index]
        gt_entities = []
        for nested_level_index in range(len(gt_labels)):  # all nested levels.
            # for nested_level_index in range(config.max_nested_level):  # todo attention this gt label!
            output_level(config.result_file, nested_level_index, config.bio_labels, gt_labels[nested_level_index], "gt")

            gt_entities, added_flag = process_one_nested_level_predict_result(config, gt_labels[nested_level_index],
                                                                              gt_entities)
            if added_flag is False:
                break
        predict_candidates = _test_one_sentence(config, model, word_ids)
        evaluate(config, predict_candidates, gt_entities)
        output_summary(config.result_file, config.test_str[test_index], config.labels, predict_candidates, gt_entities)
    # print result
    print(get_metrics(config))
    config.result_file.close()
    return


def start_test(config: Config, model: AttentionNestedNERModel):
    print("Start Testing------------------------------------------------", "\n" * 2)
    for epoch in range(config.start_test_epoch - 1, config.max_epoch):
        model = config.load_model(model, epoch)
        _test(config, model)
    return


def main():
    config = Config()
    word_dict = geniaDataset()
    model = AttentionNestedNERModel(config, word_dict).cuda() if config.cuda else AttentionNestedNERModel(config,
                                                                                                          word_dict)

    config.test_data, config.test_str, config.test_label = data_prepare(config, config.get_test_path(), word_dict)

    start_test(config, model)


if __name__ == '__main__':
    main()
# config = Config()
# # print(config.bio_labels)
# # print([i for i in range(len(config.bio_labels))])
# test_list = [1, 2, 2, 2, 3, 3, 4, 4, 0, 0, 5, 5, 6, 6, 6, 7]
# print(find_labels(test_list, config))


# main()
