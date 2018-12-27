# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2018-10-09
import queue

import numpy as np
import torch as t
import torch.nn.functional as F
from torch.autograd import Variable
import config as cfg
from model import RCNN
import utils
from utils import reg_to_bbox, calc_ious_1d, get_count_dataframe_by_confusion_matrix, denorm
from utils import first_write, lb_reg_to_bbox, repredic
from utils import non_maximum_suppression_fusion, evaluate
from arguments import TestArguments

word_embedding_dim = cfg.WORD_EMBEDDING_DIM
classes_num = cfg.CLASSES_NUM
lower_than_th_count = [0]


def load_model(test_arguments):
    rcnn = RCNN(test_arguments.pos_loss_method, test_arguments.loss_weight_lambda).cuda()
    rcnn.load_state_dict(t.load(test_arguments.model_path))
    rcnn.eval()  # dropout rate = 0
    return rcnn


def load_test_sentence(test_sentence_npz_path):
    npz = np.load(test_sentence_npz_path)
    test_sentences = npz['test_sentences']
    test_sentence_info = npz['test_sentence_info']
    test_roi = npz['test_roi']

    test_sentences = t.Tensor(test_sentences).cuda()

    return test_sentences, test_sentence_info, test_roi


def select_meaningful_bbox_regression(predict_bbox, predict_cls_score):
    argmax_softmax = np.argmax(predict_cls_score, axis=1)
    new_predict_bbox = np.zeros(
        (predict_bbox.shape[0], predict_bbox.shape[1]))  # only select the meaningful bbox regression.
    for i in range(predict_bbox.shape[1]):  # for n region
        new_predict_bbox[:, i] = predict_bbox[:, i, argmax_softmax[i]]
    return new_predict_bbox, argmax_softmax


def score_filter(predict_cls_score, score_th):
    max_score = np.max(predict_cls_score, axis=1)
    filter_indexes = np.where(max_score > score_th)
    return filter_indexes[0]


def check_regression(predict_bbox, rois, original_predict_cls_score, sentence_len):
    """Only select"""
    final_predict_bbox = int_bbox(predict_bbox, sentence_len)
    ious = calc_ious_1d(final_predict_bbox, rois)
    index = np.array([True if ious[i, i] >= 0.8 else False for i in range(len(predict_bbox))])
    original_predict_cls_score[index, 1:cfg.CLASSES_NUM + 1] -= 0
    try:
        a = predict_bbox[index], rois[index], F.softmax(t.Tensor(original_predict_cls_score[index]),
                                                        dim=1).data.cpu().numpy()
    except:
        return predict_bbox[index], rois[index], np.array([])
    return a


def _test_one_sentence(test_arguments, sentence, rois, rcnn, this_sentence_len, info):
    roi_num = rois.shape[0]
    ridx = np.zeros(roi_num).astype(int)
    original_predict_cls_score, predict_target_bbox = rcnn(sentence, rois, ridx)
    predict_cls_score = F.softmax(original_predict_cls_score, dim=1).data.cpu().numpy()  # softmax score for each row
    argmax_softmax = np.argmax(predict_cls_score, axis=1)
    if test_arguments.loss_weight_lambda == 0:
        predict_bbox = rois
        write_result(predict_bbox, argmax_softmax, np.max(predict_cls_score, axis=1),
                     test_arguments.result_output, info['gt_str'], None, None, 'TOI')
    else:
        predict_target_bbox = predict_target_bbox.data.cpu().numpy()
        if test_arguments.normalize:
            predict_target_bbox = denorm(predict_target_bbox, test_arguments)
        if test_arguments.dx_compute_method == "left_boundary":
            predict_bbox = lb_reg_to_bbox(this_sentence_len, predict_target_bbox, rois)
        else:
            predict_bbox = reg_to_bbox(this_sentence_len, predict_target_bbox, rois)

        new_predict_bbox = np.zeros(
            (predict_bbox.shape[1], predict_bbox.shape[0]))  # only select the meaningful bbox regression.
        for i in range(predict_bbox.shape[1]):  # for n region
            new_predict_bbox[i, :] = predict_bbox[:, i, argmax_softmax[i]]
        write_result(int_bbox(new_predict_bbox, this_sentence_len).astype(np.int32), argmax_softmax,
                     np.max(predict_cls_score, axis=1),
                     test_arguments.result_output, info['gt_str'], None, None, 'TOI')

        predict_bbox, rois, predict_cls_score = check_regression(new_predict_bbox, rois,
                                                                 original_predict_cls_score.data.cpu().numpy(),
                                                                 this_sentence_len)
        write_result(predict_bbox.astype(np.int32), np.argmax(predict_cls_score, axis=1),
                     np.max(predict_cls_score, axis=1),
                     test_arguments.result_output, info['gt_str'], None, None, 'after check LR')

    result_bbox = []
    result_cls = []
    drop_bbox = []
    drop_cls = []
    original_roi_ls = []
    c_score_ls = []

    if len(predict_bbox) > 0:
        # if True:
        argmax_softmax = np.argmax(predict_cls_score, axis=1)
        # predict_bbox, argmax_softmax = select_meaningful_bbox_regression(predict_bbox, predict_cls_score)

        score_indexes = score_filter(predict_cls_score, score_th=test_arguments.score_threshold)

        predict_bbox = predict_bbox[score_indexes, :]
        argmax_softmax = argmax_softmax[score_indexes]
        predict_cls_score = predict_cls_score[score_indexes]
        rois = rois[score_indexes]
        write_result(predict_bbox.astype(np.int32), argmax_softmax, np.max(predict_cls_score, axis=1),
                     test_arguments.result_output, info['gt_str'], None, None, 'after filter')
        for c in range(1, classes_num + 1):
            cls_index = np.where(argmax_softmax == c)
            if len(cls_index[0]) == 0:  # this cls type is empty
                continue
            c_cls_score = predict_cls_score[cls_index[0], c]
            c_bboxs = predict_bbox[cls_index[0], :]
            original_roi = rois[np.where(argmax_softmax == c)[0], :]
            boxes, _boxes, roi, c_score = non_maximum_suppression_fusion(c_cls_score, c_bboxs, original_roi,
                                                                         iou_threshold=test_arguments.th_nms_iou,
                                                                         score_threshold=test_arguments.score_threshold,
                                                                         info=info)
            result_bbox.extend(boxes)
            drop_bbox.extend(_boxes)
            original_roi_ls.extend(roi)
            c_score_ls.extend(c_score)
            result_cls.extend([c] * len(boxes))  # print the predict result of this sentence.
            drop_cls.extend([c] * len(_boxes))
    return np.array(result_bbox), np.array(result_cls), np.array(drop_bbox), np.array(drop_cls), np.array(
        original_roi_ls), np.array(c_score_ls)


def _test_epoch(test_arguments):
    rcnn = load_model(test_arguments)
    test_sentences, test_sentence_info, test_roi = load_test_sentence(test_arguments.test_sentence_npz_path)

    for index in range(test_sentences.size(0)):
        sentence = Variable(test_sentences[index:index + 1]).cuda()

        info = test_sentence_info[index]

        idxs = info['roi_ids']
        this_sentence_len = len(info['gt_str']) - 1
        rois = test_roi[idxs]

        write_result(None, None, None, test_arguments.result_output, info['gt_str'], info['gt_bboxs'],
                     info['gt_cls'], 'begin')

        result_bbox, result_cls, drop_bbox, drop_cls, original_rois, scores = _test_one_sentence(
            test_arguments, sentence, rois, rcnn, this_sentence_len, info=info)
        if test_arguments.loss_weight_lambda != 0:
            result_bbox = int_bbox(result_bbox, this_sentence_len)

        write_result(result_bbox.astype(np.int32), result_cls, scores, test_arguments.result_output, info['gt_str'],
                     None, None, 'after NMS')

        q = queue.Queue()
        q.put([result_bbox, result_cls, scores])
        predict_bboxes, predict_cls, score = repredic(q)

        write_result(result_bbox.astype(np.int32), result_cls, scores, test_arguments.result_output, info['gt_str'],
                     None, None, 'after delete contain')
        evaluate(predict_bboxes, predict_cls, info, test_arguments.confusion_matrix, test_arguments.th_iou_p)
    return


def write_result(predict_bboxes, predict_cls, score, outfile, gt_str, gt_bboxes, gt_cls, type):
    if type == 'begin':
        outfile.write(' '.join(gt_str) + '\ngt: ')
        for i in range(len(gt_cls)):
            outfile.write(
                cfg.LABEL[gt_cls[i] - 1] + ' [' + ' '.join(gt_str[gt_bboxes[i][0]: gt_bboxes[i][1] + 1]) + ']  ')
    else:
        outfile.write('--------------------------------------------------------\n' + type + '\n')
        for i in range(len(predict_cls)):
            if predict_cls[i] != 0:
                outfile.write(
                    cfg.LABEL[predict_cls[i] - 1] + ' [' + ' '.join(
                        gt_str[predict_bboxes[i][0]: predict_bboxes[i][1] + 1])
                    + '] ' + str(predict_bboxes[i]) + str(np.round(score[i], 3)) + '\n')
    outfile.write('\n')
    if type == 'after delete contain':
        outfile.write('\n_______________________________________\n')
    return


def int_bbox(bbox, sentence_len):
    """Integer the boundary of TOI"""
    if bbox.size > 0:
        bbox[:, 0] = np.round(bbox[:, 0], 0)
        bbox[:, 1] = np.round(bbox[:, 1], 0)
        bbox[:, 0][bbox[:, 0] < 0] = 0
        bbox[:, 1][bbox[:, 1] > sentence_len] = sentence_len
    return bbox


def _test(test_arguments, all_csv_result):
    for model_epoch in range(test_arguments.min_test_epoch, test_arguments.max_test_epoch + 1):
        test_arguments.initialize_confusion_matrix()
        test_arguments.model_path = test_arguments.get_model_path(model_epoch)
        test_arguments.test_sentence_npz_path = test_arguments.get_test_npz_path()
        _test_epoch(test_arguments)
        print("Not even classify right", utils.not_detect_ls)
        print("NMS Drop bbox", utils.nms_mis_hit)
        print("IOU lower than th", utils.iou_lower_hit)
        utils.not_detect_ls = [0] * cfg.CLASSES_NUM
        utils.nms_mis_hit = [0] * cfg.CLASSES_NUM
        utils.iou_lower_hit = [0] * cfg.CLASSES_NUM

        lower_than_th_count[0] = 0
        print(test_arguments.confusion_matrix.astype(np.int32))

        test_arguments.confusion_matrix = test_arguments.confusion_matrix.astype(np.int32).tolist()
        test_arguments.data_frame = get_count_dataframe_by_confusion_matrix(test_arguments.confusion_matrix)
        test_arguments.write_one_result(all_csv_result)
        print(test_arguments.data_frame)
        print('contain:')
        print(str(test_arguments.contain_matrix))
    return


def main():
    mode = "test"  # test  # debug

    pos_loss_method = "mse"  # "mse"  # "smoothL1"
    th_train_iou = 0.8  # 0.8
    norm = True  # False
    min_test_epoch = 32
    max_test_epoch = 32
    loss_weight_lambda_ls = [0]  # [0.1, 0.2, 0.5, 1, 2, 5, 10]
    prevent_overfitting_method = "L2 Regu"  # "L2 Regu" # "Dropout"
    partial_l2 = False
    dx_compute_method = "centre"

    th_score = 0.8

    test_arguments = TestArguments(norm, pos_loss_method, th_train_iou, min_test_epoch, max_test_epoch,
                                   dx_compute_method=dx_compute_method,
                                   score_threshold=th_score, partial_l2_penalty=partial_l2,
                                   prevent_overfitting_method=prevent_overfitting_method, mode=mode)
    test_arguments.show_arguments()
    write_result_path = test_arguments.get_write_result_path()

    all_csv_result = open(write_result_path, "w")
    first_write(all_csv_result)
    th_nms_iou_ls = [0.01]
    th_iou_p_ls = [1]
    for loss_weight_lambda in loss_weight_lambda_ls:
        for th_iou_p in th_iou_p_ls:
            for th_nms_iou in th_nms_iou_ls:
                test_arguments.loss_weight_lambda = loss_weight_lambda
                test_arguments.th_iou_p = th_iou_p
                test_arguments.th_nms_iou = th_nms_iou
                _test(test_arguments, all_csv_result)


if __name__ == '__main__':
    main()
