# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2018-10-09

import numpy as np
import torch as t
import torch.nn.functional as F
from torch.autograd import Variable
import config as cfg
from model import RCNN
import utils
from utils import reg_to_bbox, non_maximum_suppression, evaluate, get_count_dataframe_by_confusion_matrix, denorm
from utils import first_write, lb_reg_to_bbox
from utils import non_maximum_suppression_all_regression, non_maximum_suppression_l
from arguments import TestAruguments

word_embedding_dim = cfg.WORD_EMBEDDING_DIM
classes_num = cfg.CLASSES_NUM
th_score = cfg.TH_SCORE
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


def select_meaningful_bbox_regression(pred_bbox, pred_cls_score):
    argmax_softmax = np.argmax(pred_cls_score, axis=1)
    new_pred_bbox = np.zeros((pred_bbox.shape[0], pred_bbox.shape[1]))  # only select the meaningful bbox regression.
    for i in range(pred_bbox.shape[1]):  # for n region
        new_pred_bbox[:, i] = pred_bbox[:, i, argmax_softmax[i]]
    return new_pred_bbox, argmax_softmax


def score_filter(pred_cls_score, score_th):
    max_score = np.max(pred_cls_score, axis=1)
    filter_indexs = np.where(max_score > score_th)
    return filter_indexs[0]


def _test_one_sentence(file_before, test_arguments, sentence, rois, rcnn, fold_index, this_sentence_len,
                       info):
    roi_num = rois.shape[0]
    ridx = np.zeros(roi_num).astype(int)
    pred_cls_score, pred_tbbox = rcnn(sentence, rois, ridx)
    pred_cls_score = F.softmax(pred_cls_score, dim=1).data.cpu().numpy()  # softmax score for each row
    pred_tbbox = pred_tbbox.data.cpu().numpy()
    if test_arguments.normalize:
        pred_tbbox = denorm(pred_tbbox, fold_index, test_arguments)
    if test_arguments.dx_compute_method == "left_boundary":
        pred_bbox = lb_reg_to_bbox(this_sentence_len, pred_tbbox, rois)
    else:
        pred_bbox = reg_to_bbox(this_sentence_len, pred_tbbox, rois)
    original_pred_bbox = pred_bbox.copy()
    pred_bbox, argmax_softmax = select_meaningful_bbox_regression(pred_bbox, pred_cls_score)
    score_indexs = score_filter(pred_cls_score, score_th=test_arguments.score_threshold)
    pred_bbox = pred_bbox[:, score_indexs]
    argmax_softmax = argmax_softmax[score_indexs]
    pred_cls_score = pred_cls_score[score_indexs]
    rois = rois[score_indexs]

    output_bbox = pred_bbox.T.copy()
    output_result_cls = np.argmax(pred_cls_score, axis=1).copy()
    int_bbox(output_bbox, len(info["gt_str"]))
    output_file_result(file_before, output_bbox, output_result_cls, rois, info, np.max(pred_cls_score, axis=1))
    # if test_arguments.output_flow:
    #     output_detect_result(output_bbox, output_result_cls, rois, info, np.max(pred_cls_score, axis=1))

    # if test_arguments.output_flow:
    #     output_detect_result(result_bbox, result_cls, original_rois, info, scores)

    # pred bbox shape is 2 * n_roi
    result_bbox = []
    result_cls = []
    drop_bbox = []
    drop_cls = []
    original_roi_ls = []
    c_score_ls = []
    for c in range(1, classes_num + 1):
        cls_index = np.where(argmax_softmax == c)
        if len(cls_index[0]) == 0:  # this cls type is empty
            continue
        c_cls_score = pred_cls_score[cls_index[0], c]
        c_bboxs = pred_bbox[:, cls_index[0]].T
        original_roi = rois[np.where(argmax_softmax == c)[0], :]
        if c == 1:
            boxes, _boxes, roi, c_score = non_maximum_suppression(c_cls_score, c_bboxs, original_roi,
                                                                  iou_threshold=0.01,
                                                                  score_threshold=test_arguments.score_threshold,
                                                                  info=info)
        else:
            boxes, _boxes, roi, c_score = non_maximum_suppression_l(c_cls_score, c_bboxs, original_roi,
                                                                    iou_threshold=test_arguments.th_nms_iou,
                                                                    score_threshold=test_arguments.score_threshold,
                                                                    info=info)
        result_bbox.extend(boxes)
        drop_bbox.extend(_boxes)
        original_roi_ls.extend(roi)
        c_score_ls.extend(c_score)
        result_cls.extend([c] * len(boxes))  # print the predict result of this sentence.
        drop_cls.extend([c] * len(_boxes))
    # output_file_result(file_after, result_bbox, result_cls, original_roi_ls, info, np.max(pred_cls_score, axis=1))

    return np.array(result_bbox), np.array(result_cls), np.array(drop_bbox), np.array(drop_cls), np.array(
        original_roi_ls), np.array(c_score_ls)


def output_file_result(file, result_bbox, result_cls, original_rois, info, scores):
    str_sentence = info["gt_str"]
    for roi_index in range(len(result_bbox)):
        if result_cls[roi_index] > 0:
            file.write(
                "原本ROI:   " + " ".join(
                    str_sentence[original_rois[roi_index][0]:original_rois[roi_index][1] + 1]) + "\n")
            file.write("Regression之后:   " + " ".join(
                str_sentence[int(result_bbox[roi_index][0]):int(result_bbox[roi_index][1]) + 1]) + "\n")
            file.write("类别:   " + cfg.LABEL[result_cls[roi_index] - 1] + " \n")
            file.write("分数:   " + str(scores[roi_index]) + "\n")
            file.write("\n\n")
        # file.write(
        #     "原本ROI:   " + " ".join(
        #         str_sentence[original_rois[roi_index][0]:original_rois[roi_index][1] + 1]) + "\n")
        # file.write("Regression之后:   " + " ".join(
        #     str_sentence[int(result_bbox[roi_index][0]):int(result_bbox[roi_index][1]) + 1]) + "\n")
        # if result_cls[roi_index] > 0:
        #     file.write("类别:   " + cfg.LABEL[result_cls[roi_index] - 1] + "\n")
        # else:
        #     file.write("类别:   " + "BG" + "\n")
        # file.write("分数:   " + str(scores[roi_index]) + "\n")
        # file.write("\n\n")
    # file.close()
    return


def _test_one_sentence_all_regression(test_arguments, sentence, rois, rcnn, fold_index, this_sentence_len, info):
    roi_num = rois.shape[0]
    ridx = np.zeros(roi_num).astype(int)
    pred_cls_score, pred_tbbox = rcnn(sentence, rois, ridx)
    pred_cls_score = F.softmax(pred_cls_score, dim=1).data.cpu().numpy()  # softmax score for each row
    pred_tbbox = pred_tbbox.data.cpu().numpy()
    if test_arguments.normalize:
        pred_tbbox = denorm(pred_tbbox, fold_index, test_arguments)
    if test_arguments.dx_compute_method == "left_boundary":
        pred_bbox = lb_reg_to_bbox(this_sentence_len, pred_tbbox, rois)
    else:
        pred_bbox = reg_to_bbox(this_sentence_len, pred_tbbox, rois)

    result_bbox = []
    result_cls = []
    drop_bbox = []
    drop_cls = []
    original_roi_ls = []
    c_score_ls = []
    for c in range(1, classes_num + 1):
        c_cls_score = pred_cls_score[:, c]
        c_bboxs = pred_bbox[:, :, c].T
        boxes, _boxes, roi, c_score = non_maximum_suppression_all_regression(c_cls_score, c_bboxs, rois.copy(),
                                                                             iou_threshold=test_arguments.th_nms_iou,
                                                                             score_threshold=test_arguments.score_threshold,
                                                                             info=info)
        result_bbox.extend(boxes)
        drop_bbox.extend(_boxes)
        original_roi_ls.extend(roi)
        c_score_ls.extend(c_score)
        result_cls.extend([c] * len(boxes))  # print the predict result of this sentence.
        drop_cls.extend([c] * len(_boxes))

    if len(result_cls) == 0:  # if the sentence without detecting anything!, we will lower the score criterion
        lower_than_th_count[0] += 1
        drop_bbox = []
        drop_cls = []
        for c in range(1, classes_num + 1):
            c_sc = pred_cls_score[:, c]
            c_bboxs = pred_bbox[:, :, c].T
            boxes, _boxes, roi, c_score = non_maximum_suppression(c_sc, c_bboxs, rois.copy(),
                                                                  iou_threshold=test_arguments.th_nms_iou,
                                                                  score_threshold=test_arguments.score_threshold / 2,
                                                                  info=info)
            result_bbox.extend(boxes)
            drop_bbox.extend(_boxes)
            original_roi_ls.extend(roi)
            c_score_ls.extend(c_score)
            result_cls.extend([c] * len(boxes))
            drop_cls.extend([c] * len(_boxes))
        # result_bbox = result_bbox[:1]
        # result_cls = result_cls[:1]
        # drop_bbox += result_bbox[1:]
        # drop_cls += result_cls[1:]
        # original_roi_ls = original_roi_ls[:1]
        # c_score_ls = c_score_ls[:1]

    return np.array(result_bbox), np.array(result_cls), np.array(drop_bbox), np.array(drop_cls), np.array(
        original_roi_ls), np.array(c_score_ls)


def _test_epoch(file_before, file_after, test_arguments, fold_index):
    rcnn = load_model(test_arguments)
    test_sentences, test_sentence_info, test_roi = load_test_sentence(test_arguments.test_sentence_npz_path)

    sentence_num = test_sentences.size(0)

    for i in range(sentence_num):
        sentence = Variable(test_sentences[i:i + 1]).cuda()

        info = test_sentence_info[i]

        idxs = info['roi_ids']
        this_sentence_len = len(info['gt_str'])
        rois = test_roi[idxs]

        result_bbox, result_cls, drop_bbox, drop_cls, original_rois, scores = _test_one_sentence(
            file_before,
            test_arguments,
            sentence, rois,
            rcnn,
            fold_index,
            this_sentence_len,
            info=info)
        # todo modify the sentence result only the max softmax result.
        result_bbox = int_bbox(result_bbox, this_sentence_len)
        drop_bbox = int_bbox(drop_bbox, this_sentence_len)
        if test_arguments.output_flow:
            output_detect_result(result_bbox, result_cls, original_rois, info, scores)
        output_file_result(file_after, result_bbox, result_cls, original_rois, info, scores)

        evaluate(result_bbox, result_cls, drop_bbox, drop_cls, info, test_arguments.confusion_matrix,
                 test_arguments.th_iou_p)
    return


def int_bbox(bbox, sentence_len):
    if bbox.size > 0:
        bbox[:, 0] = np.round(bbox[:, 0], 0)
        bbox[:, 1] = np.round(bbox[:, 1], 0)
        bbox[:, 0][bbox[:, 0] < 0] = 0
        bbox[:, 1][bbox[:, 1] > sentence_len] = sentence_len
    return bbox


def output_detect_result(result_bbox, result_cls, original_rois, info, scores):
    str_sentence = info["gt_str"]
    for roi_index in range(len(result_bbox)):
        if result_cls[roi_index] > 0:
            print("原本ROI:   ", str_sentence[original_rois[roi_index][0]:original_rois[roi_index][1] + 1])
            print("Regression之后:   ", str_sentence[int(result_bbox[roi_index][0]):int(result_bbox[roi_index][1]) + 1])
            print("类别:   ", cfg.LABEL[result_cls[roi_index] - 1])
            print("分数:   ", scores[roi_index])
            print()
    return


def _test_k_fold(file_before, file_after, test_arguments, all_csv_result):
    for model_epoch in range(test_arguments.min_test_epoch, test_arguments.max_test_epoch + 1):
        test_arguments.initialize_confusion_matrix()
        for fold_index in range(test_arguments.fold_k):
            test_arguments.model_path = test_arguments.get_model_path(fold_index, model_epoch)
            test_arguments.test_sentence_npz_path = test_arguments.get_test_npz_path(fold_index)
            _test_epoch(file_before, file_after, test_arguments, fold_index)
        print("Not even classify right", utils.not_detect_ls)
        print("NMS Drop bbox", utils.nms_mis_hit)
        print("IOU lower than th", utils.iou_lower_hit)
        utils.not_detect_ls = [0] * cfg.CLASSES_NUM
        utils.nms_mis_hit = [0] * cfg.CLASSES_NUM
        utils.iou_lower_hit = [0] * cfg.CLASSES_NUM

        lower_than_th_count[0] = 0
        test_arguments.confusion_matrix = test_arguments.confusion_matrix.astype(np.int32).tolist()
        test_arguments.data_frame = get_count_dataframe_by_confusion_matrix(test_arguments.confusion_matrix)
        # print(utils.nms_mis_hit)
        # utils.nms_mis_hit = [0] * 7
        test_arguments.write_one_result(all_csv_result)
        print(test_arguments.data_frame)
    return


def main():
    pos_loss_method = "mse"  # "mse"
    th_train_iou = 0.6  # 0.8
    norm = True  # False
    min_test_epoch = 31
    max_test_epoch = 40
    loss_weight_lambda = 1.0
    prevent_overfitting_method = "Dropout"  # "L2 Regu" # "Dropout"
    partial_l2 = False
    dx_compute_method = "centre"
    output_flow = False
    th_score = 0.45
    test_arguments = TestAruguments(norm, pos_loss_method, th_train_iou, min_test_epoch, max_test_epoch,
                                    dx_compute_method=dx_compute_method, output_flow=output_flow,
                                    score_threshold=th_score,
                                    loss_weight_lambda=loss_weight_lambda, partial_l2_penalty=partial_l2,
                                    prevent_overfitting_method=prevent_overfitting_method)
    test_arguments.show_arguments()
    write_result_path = test_arguments.get_write_result_path()

    all_csv_result = open(write_result_path, "w")
    first_write(all_csv_result)
    th_nms_iou_ls = [0]
    th_iou_p_ls = [0.6, 0.8, 1]
    for th_iou_p in th_iou_p_ls:
        for th_nms_iou in th_nms_iou_ls:
            file_before = open("Debug Before NMS.txt", "w")
            file_after = open("Debug After NMS.txt", "w")
            test_arguments.th_nms_iou = th_nms_iou
            test_arguments.th_iou_p = th_iou_p
            _test_k_fold(file_before, file_after, test_arguments, all_csv_result)


if __name__ == '__main__':
    main()
