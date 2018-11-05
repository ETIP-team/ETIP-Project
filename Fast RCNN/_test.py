# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2018-10-09


import os
import numpy as np
import torch as t
import torch.nn.functional as F
from torch.autograd import Variable
import config as cfg
from model import RCNN
from utils import reg_to_bbox, non_maximum_suppression, evaluate, get_count_dataframe_by_confusion_matrix, denorm
from utils import first_write, write_one_result
from arguments import TestAruguments

word_embedding_dim = cfg.WORD_EMBEDDING_DIM
classes_num = cfg.CLASSES_NUM
th_score = cfg.TH_SCORE


def load_model(test_arguments):
    rcnn = RCNN(test_arguments.pos_loss_method, test_arguments.loss_weight_lambda).cuda()
    rcnn.load_state_dict(t.load(test_arguments.model_path))
    return rcnn


def load_test_sentence(test_sentence_npz_path):
    npz = np.load(test_sentence_npz_path)
    test_sentences = npz['test_sentences']
    test_sentence_info = npz['test_sentence_info']
    test_roi = npz['test_roi']

    test_sentences = t.Tensor(test_sentences).cuda()

    return test_sentences, test_sentence_info, test_roi


def _test_one_sentence(test_arguments, sentence, rois, rcnn, fold_index):
    roi_num = rois.shape[0]
    ridx = np.zeros(roi_num).astype(int)
    pred_cls_score, pred_tbbox = rcnn(sentence, rois, ridx)
    pred_cls_score = F.softmax(pred_cls_score, dim=1).data.cpu().numpy()  # softmax score for each row
    pred_tbbox = pred_tbbox.data.cpu().numpy()
    if test_arguments.normalize:
        pred_tbbox = denorm(pred_tbbox, fold_index, test_arguments.th_train_iou)

    pred_bbox = reg_to_bbox(sentence.size(2), pred_tbbox, rois)

    result_bbox = []
    result_cls = []
    for c in range(1, classes_num + 1):
        c_cls_score = pred_cls_score[:, c]
        c_bboxs = pred_bbox[:, :, c].T

        boxes = non_maximum_suppression(c_cls_score, c_bboxs, iou_threshold=test_arguments.th_nms_iou,
                                        score_threshold=test_arguments.score_threshold)
        result_bbox.extend(boxes)
        result_cls.extend([c] * len(boxes))  # print the predict result of this sentence.

    if len(result_cls) == 0:  # if the sentence without detecting anything!, we will lower the score criterion
        for c in range(1, classes_num + 1):
            c_sc = pred_cls_score[:, c]
            c_bboxs = pred_bbox[:, :, c].T
            boxes = non_maximum_suppression(c_sc, c_bboxs, iou_threshold=test_arguments.th_nms_iou,
                                            score_threshold=test_arguments.score_threshold / 2)
            result_bbox.extend(boxes)
            result_cls.extend([c] * len(boxes))
        result_bbox = result_bbox[:1]
        result_cls = result_cls[:1]

    # print(result_cls)
    return np.array(result_bbox), np.array(result_cls)


def _test_epoch(test_arguments, fold_index):
    rcnn = load_model(test_arguments)
    test_sentences, test_sentence_info, test_roi = load_test_sentence(test_arguments.test_sentence_npz_path)

    sentence_num = test_sentences.size(0)

    perm = np.random.permutation(sentence_num)

    for i in range(sentence_num):
        pi = perm[i]
        sentence = Variable(test_sentences[pi:pi + 1]).cuda()

        info = test_sentence_info[pi]
        idxs = info['roi_ids']
        rois = test_roi[idxs]

        result_bbox, result_cls = _test_one_sentence(test_arguments, sentence, rois, rcnn, fold_index)

        evaluate(result_bbox, result_cls, info, test_arguments.confusion_matrix, test_arguments.th_iou_p)

    return


def _test_k_fold(test_arguments, all_csv_result):
    for model_epoch in range(test_arguments.min_test_epoch, test_arguments.max_test_epoch + 1):
        test_arguments.initialize_confusion_matrix()
        for fold_index in range(test_arguments.fold_k):
            test_arguments.model_path = test_arguments.get_model_path(fold_index, model_epoch)
            test_arguments.test_sentence_npz_path = test_arguments.get_test_npz_path(fold_index)
            _test_epoch(test_arguments, fold_index)

        test_arguments.confusion_matrix = test_arguments.confusion_matrix.astype(np.int32).tolist()
        test_arguments.data_frame = get_count_dataframe_by_confusion_matrix(test_arguments.confusion_matrix)

        test_arguments.write_one_result(all_csv_result)
        print(test_arguments.data_frame)
    return


def main():
    pos_loss_method = "mse"  # "mse"
    th_train_iou = 0.6  # 0.8
    norm = True  # Flase
    min_test_epoch = 31
    max_test_epoch = 40
    loss_weight_lambda = 2.0
    prevent_overfitting_method = "L2 Norm"

    test_arguments = TestAruguments(norm, pos_loss_method, th_train_iou, min_test_epoch, max_test_epoch,
                                    loss_weight_lambda=loss_weight_lambda,
                                    prevent_overfitting_method=prevent_overfitting_method)
    test_arguments.show_arguments()
    write_result_path = test_arguments.get_write_result_path()

    all_csv_result = open(write_result_path, "w")
    first_write(all_csv_result)
    th_nms_iou_ls = [0.01]
    th_iou_p_ls = [0.6, 0.8, 1]  # [1]
    for th_iou_p in th_iou_p_ls:
        for th_nms_iou in th_nms_iou_ls:
            test_arguments.th_nms_iou = th_nms_iou
            test_arguments.th_iou_p = th_iou_p

            _test_k_fold(test_arguments, all_csv_result)


if __name__ == '__main__':
    main()
