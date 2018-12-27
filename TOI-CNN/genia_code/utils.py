# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2018-10-06

import jieba
import numpy as np
import pandas as pd
import config as cfg

ignore_str = "，。,；：:…． "
split_chars = '[，。,；：:…．;]'

word_embedding_dim = cfg.WORD_EMBEDDING_DIM
sentence_length = cfg.SENTENCE_LENGTH
empty_wv = np.zeros((1, word_embedding_dim))
classes_num = cfg.CLASSES_NUM

contain_matrix = [[59, 0, 1, 195, 0],
                  [14, 0, 0, 52., 0],
                  [0, 0, 43, 2, 4],
                  [21, 3, 5, 209, 0],
                  [0, 0, 16, 14, 22]]

not_hit = [[] for i in range(classes_num)]
hit = [0]
# 12-2
norm_info = {
    "0.6": {
        "centre": {
            "all_mean": [[0.016632804349019824, -0.11897047709648585]],
            "all_deviation": [[1.0961096534213135, 0.30693177648954845]]},
    },
    "0.7": {
        "centre": {
            "all_mean": [[0.007462601542650761, -0.066652877916399]],
            "all_deviation": [[0.7630193951879333, 0.20092074140625282]]
        }
    },
    "0.8": {
        "centre": {
            "all_mean": [[0.00042789223454833597, -0.02895624176034832]],
            "all_deviation": [[0.4650596693405339, 0.1276511062431574]]
        }
    }
}

nms_mis_hit = [0] * cfg.CLASSES_NUM
not_detect_ls = [0] * cfg.CLASSES_NUM
iou_lower_hit = [0] * cfg.CLASSES_NUM


def first_write(all_csv_result):
    row = cfg.LABEL + ["Overall"]
    all_csv_result.write(",,,,,,,")
    for row_info in row[:-1]:
        all_csv_result.write(row_info + ",,,")
    all_csv_result.write(row[-1] + "\n")


def write_one_result(all_csv_result, th_train_iou, th_iou_nms, th_iou_p, temp_pandas):
    data_lists = temp_pandas.values.tolist()
    print("\nTh_train_iou:  ", th_train_iou, "th_iou_nms", th_iou_nms, "th_iou_p", th_iou_p)
    all_csv_result.write(
        "th_train_iou= " + str(th_train_iou) + ", th_nms= " + str(th_iou_nms) + ", th_p= " + str(th_iou_p) + ",,")
    for d_l in data_lists:
        for d in d_l:
            all_csv_result.write(str(d))
            all_csv_result.write(",")
    all_csv_result.write("\n")


def denorm(pred_bbox, test_arguments):
    all_deviation = norm_info[str(test_arguments.th_train_iou)][test_arguments.dx_compute_method]["all_deviation"]
    all_mean = norm_info[str(test_arguments.th_train_iou)][test_arguments.dx_compute_method]["all_mean"]
    pred_bbox[:, :, 0] = (pred_bbox[:, :, 0] * all_deviation[0][0]) + all_mean[0][0]
    pred_bbox[:, :, 1] = (pred_bbox[:, :, 1] * all_deviation[0][1]) + all_mean[0][1]
    return pred_bbox


def lb_bbox_transform_1d(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 1] - ex_rois[:, 0]
    ex_lb_x = ex_rois[:, 0]

    gt_widths = gt_rois[:, 1] - gt_rois[:, 0]
    gt_lb_x = gt_rois[:, 0]

    # targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dx = (gt_lb_x - ex_lb_x)
    targets_dw = np.log(gt_widths / ex_widths)

    targets = np.array([targets_dx, targets_dw]).T
    return targets


def bbox_transform_1d(ex_rois, gt_rois):  # modify in 11-16
    ex_widths = ex_rois[:, 1] - ex_rois[:, 0] + 1
    ex_ctr_x = (ex_rois[:, 1] + ex_rois[:, 0]) / 2

    gt_widths = gt_rois[:, 1] - gt_rois[:, 0] + 1
    gt_ctr_x = (gt_rois[:, 1] + gt_rois[:, 0]) / 2

    targets_dx = gt_ctr_x - ex_ctr_x
    targets_dw = np.log(gt_widths / ex_widths)

    targets = np.array([targets_dx, targets_dw]).T
    return targets


def calc_intersection_union(ex_rois, gt_rois):  # modify  in 11-16
    ex_width = ex_rois[:, 1] - ex_rois[:, 0] + 1
    gt_width = gt_rois[:, 1] - gt_rois[:, 0] + 1

    area_sum = ex_width.reshape((-1, 1)) + gt_width.reshape((1, -1))

    lb = np.maximum(ex_rois[:, 0].reshape((-1, 1)), gt_rois[:, 0].reshape((1, -1)))
    rb = np.minimum(ex_rois[:, 1].reshape((-1, 1)) + 1, gt_rois[:, 1].reshape((1, -1)) + 1)

    area_intersection = np.maximum(rb - lb, 0)
    area_union = area_sum - area_intersection

    return area_intersection, area_union


def calc_ious_1d(ex_rois, gt_rois):  # modify  in 11-16
    ex_width = ex_rois[:, 1] - ex_rois[:, 0] + 1
    gt_width = gt_rois[:, 1] - gt_rois[:, 0] + 1

    area_sum = ex_width.reshape((-1, 1)) + gt_width.reshape((1, -1))

    lb = np.maximum(ex_rois[:, 0].reshape((-1, 1)), gt_rois[:, 0].reshape((1, -1)))
    rb = np.minimum(ex_rois[:, 1].reshape((-1, 1)) + 1, gt_rois[:, 1].reshape((1, -1)) + 1)

    area_intersection = np.maximum(rb - lb, 0)
    area_union = area_sum - area_intersection
    ious = area_intersection / area_union
    return ious


def list_text_seg(ls):
    """input the list of strings you want to cut by jieba
        return the list, every item contains the one list of original string"""
    for i in range(len(ls)):
        ls1 = []
        x = jieba.cut(ls[i])
        for a in x:
            ls1.append(a)
        ls[i] = ls1
    return ls


def str2wv(ls, word_vector_model):
    arr = empty_wv
    for i in ls:
        try:
            arr = np.vstack((arr, word_vector_model[i]))
        except KeyError:
            arr = np.vstack((arr, empty_wv))
            # print(j, "can not be found in word2vec dictionary")
    arr = np.delete(arr, 0, axis=0)
    row, column = arr.shape
    try:
        arr = np.pad(arr, ((0, sentence_length - row), (0, 0)), 'constant', constant_values=0)
    except Exception as e:
        print(ls)
        exit("Above sentence length is " + str(len(ls)) + " which was too long")
    return arr


# def list2str(ls):
#     return "".join(ls)
#
#
# def txt_split(text_str):
#     """Input: one paragraph.
#         Return: sentence word cut list.
#         like: [["w1", "w2", "w3"], ["w1", "w2"]].
#         For each one sentence in this paragraph, we return the word segmentation of one sentence.
#     """
#     ls_seg_sentences = []
#     if word_segmentation_method == "jieba":
#         for item in [sentence for sentence in re.split(split_chars, text_str) if len(sentence) > 0]:
#             ls_seg_sentences.append([seg for seg in jieba.cut(item)])
#     elif word_segmentation_method == "nlpir":
#         for item in [sentence for sentence in re.split(split_chars, text_str) if len(sentence) > 0]:
#             ls_seg_sentences.append([seg for seg, flag in pynlpir.segment(item)])
#     else:
#         raise KeyError("Word segmentation method was invalid")
#     return ls_seg_sentences


def lb_reg_to_bbox(sentence_length, reg, box):  # use x left boundary
    bbox_width = box[:, 1] - box[:, 0]  # Region length
    bbox_lb_x = box[:, 0]  # x left boundary

    bbox_width = bbox_width[:, np.newaxis]
    bbox_lb_x = bbox_lb_x[:, np.newaxis]

    out_lb_x = reg[:, :, 0] + bbox_lb_x

    out_width = bbox_width * np.exp(reg[:, :, 1])

    return np.array([
        np.maximum(0, bbox_lb_x - 0 * out_width),
        np.minimum(sentence_length, out_lb_x + 1 * out_width), ])


def reg_to_bbox(sentence_length, reg, box):
    bbox_width = box[:, 1] - box[:, 0] + 1  # Region length
    bbox_ctr_x = (box[:, 1] + box[:, 0]) / 2  # x coordinate   # Region coordinate

    bbox_width = bbox_width[:, np.newaxis]
    bbox_ctr_x = bbox_ctr_x[:, np.newaxis]

    # out_ctr_x = reg[:, :, 0] * bbox_width + bbox_ctr_x
    out_ctr_x = reg[:, :, 0] + bbox_ctr_x

    out_width = bbox_width * np.exp(reg[:, :, 1])

    return np.array([
        np.maximum(0, out_ctr_x - 0.5 * (out_width - 1)),
        np.minimum(sentence_length, out_ctr_x + 0.5 * (out_width - 1))])


def non_maximum_suppression_fusion(scores, bboxs, original_roi, iou_threshold=0.5, score_threshold=0.6, info=None,
                                   return_all_flag=True):
    roi_num = scores.shape[0]
    sentence_length = len(info['gt_str'])
    bboxs_length = bboxs[:, 1] - bboxs[:, 0] + 1
    SLR = bboxs_length / sentence_length
    sorted_length_indexes = np.argsort(bboxs_length)[::-1]
    sorted_score_indexes = np.argsort(scores)[::-1]

    lenth_rank = np.zeros(roi_num)
    score_rank = np.zeros(roi_num)
    for i in range(roi_num):
        lenth_rank[sorted_length_indexes[i]] = i
        score_rank[sorted_score_indexes[i]] = i
    mix_rank = [lenth_rank[i] * SLR[i] + score_rank[i] * (1 - SLR[i]) for i in range(roi_num)]
    sorted_mix_indexes = np.argsort(mix_rank)

    bboxs = bboxs[sorted_mix_indexes, :]
    ious = calc_ious_1d(bboxs, bboxs)

    result = []
    drop = []
    result_index = []

    for i in range(roi_num):
        if i == 0 or ious[i, result_index].max() < iou_threshold:  # rectify in 10-29
            result.append(bboxs[i])
            result_index.append(i)
        else:
            drop.append(bboxs[i])
    if return_all_flag:
        return result, drop, original_roi[sorted_score_indexes[result_index]], scores[
            sorted_score_indexes[result_index]]
    else:
        return result


def non_maximum_suppression_l(scores, bboxs, original_roi, iou_threshold=0.5, score_threshold=0.6, info=None,
                              return_all_flag=True):
    roi_num = scores.shape[0]
    sorted_score_indexes = np.argsort(bboxs[:, 1] - bboxs[:, 0])[::-1]  # sort by length.
    bboxs = bboxs[sorted_score_indexes, :]
    ious = calc_ious_1d(bboxs, bboxs)

    result = []
    drop = []
    result_index = []

    for i in range(roi_num):
        if i == 0 or ious[i, result_index].max() < iou_threshold:  # rectify in 10-29
            result.append(bboxs[i])
            result_index.append(i)
        else:
            drop.append(bboxs[i])
    if return_all_flag:
        return result, drop, original_roi[sorted_score_indexes[result_index]], scores[
            sorted_score_indexes[result_index]]
    else:
        return result


def non_maximum_suppression(scores, bboxs, original_roi, iou_threshold=0.5, score_threshold=0.6, info=None,
                            return_all_flag=True):
    roi_num = scores.shape[0]
    sorted_score_indexes = np.argsort(scores)[::-1]
    bboxs = bboxs[sorted_score_indexes, :]
    ious = calc_ious_1d(bboxs, bboxs)

    result = []
    drop = []
    result_index = []

    for i in range(roi_num):
        # if bboxs[i][0] != bboxs[i][1] and (
        #         len(result_index) == 0 or ious[i, result_index].max() < iou_threshold):
        if len(result_index) == 0 or ious[i, result_index].max() < iou_threshold:
            result.append(bboxs[i])
            result_index.append(i)
        else:
            drop.append(bboxs[i])
    if return_all_flag:
        return result, drop, original_roi[sorted_score_indexes[result_index]], scores[
            sorted_score_indexes[result_index]]
    else:
        return result


def non_maximum_suppression_all_regression(scores, bboxs, original_roi, iou_threshold=0.5, score_threshold=0.6,
                                           info=None, return_all_flag=True):
    roi_num = scores.shape[0]
    # sort the roi score and get it's index from big to small
    sorted_score_indexes = np.argsort(scores)[::-1]
    score_right_boundary = 0
    while score_right_boundary < roi_num and scores[sorted_score_indexes[score_right_boundary]] >= score_threshold:
        score_right_boundary += 1
    if score_right_boundary == 0:
        return [], [], [], [] if return_all_flag else []
    sorted_score_indexes = sorted_score_indexes[:score_right_boundary]
    if len(sorted_score_indexes) > 2:
        wait = True
    bboxs = bboxs[sorted_score_indexes, :]
    ious = calc_ious_1d(bboxs, bboxs)

    result = []
    drop = []
    result_index = []

    for i in range(score_right_boundary):
        if i == 0 or ious[i, result_index].max() < iou_threshold:  # rectify in 10-29
            result.append(bboxs[i])
            result_index.append(i)
        else:
            drop.append(bboxs[i])
    if return_all_flag:
        return result, drop, original_roi[sorted_score_indexes[result_index]], scores[
            sorted_score_indexes[result_index]]
    else:
        return result


def check_contain(bbox1, bbox2):
    """if bbox1 contain bbox2, return True"""
    x1, y1 = bbox1
    x2, y2 = bbox2
    x = min(x1, x2)
    y = max(y1, y2)
    return x == x1 and y == y1


def check_overlap(bbox1, bbox2):
    x1, y1 = bbox1
    x2, y2 = bbox2
    x = max(x1, x2)
    y = min(y1, y2)
    return x <= y


def repredic(q):
    pred_bboxes, pred_cls, score = q.get()
    len_bbox = len(pred_bboxes)
    if len_bbox == 1 or len_bbox == 0:
        return pred_bboxes, pred_cls, score
    isTrue = True
    for i in range(len(pred_bboxes)):
        for j in range(i + 1, len(pred_bboxes)):
            if check_contain(pred_bboxes[i], pred_bboxes[j]):
                if contain_matrix[pred_cls[i] - 1][pred_cls[j] - 1] == 0:
                    isTrue = False
                    break
            elif check_contain(pred_bboxes[j], pred_bboxes[i]):
                if contain_matrix[pred_cls[j] - 1][pred_cls[i] - 1] == 0:
                    isTrue = False
                    break
            elif check_overlap(pred_bboxes[i], pred_bboxes[j]):
                isTrue = False
                break
        if not isTrue:
            break
    if isTrue:
        return pred_bboxes, pred_cls, score
    sorted_index = np.argsort(score)
    pred_bboxes = pred_bboxes[sorted_index, :]
    pred_cls = pred_cls[sorted_index]
    score = score[sorted_index]
    for i in range(len_bbox):
        index = np.hstack([np.arange(0, i), np.arange(i + 1, len_bbox)])
        q.put([pred_bboxes[index], pred_cls[index], score[index]])
    return repredic(q)


def evaluate(pred_bboxes, pred_cls, info, confusion_matrix, th_iou=0.6):
    assert len(pred_bboxes) == len(pred_cls)
    gt_bboxes = info["gt_bboxs"]
    gt_cls = info["gt_cls"]
    gt_cls_hit_ls = [0 for i in range(len(gt_cls))]
    pre_cls_hit_ls = [0 for i in range(len(gt_cls))]

    if len(pred_cls) > 0:
        ious = calc_ious_1d(pred_bboxes, gt_bboxes)
        for pred_index in range(len(pred_cls)):
            if th_iou != 1:
                threshold = th_iou if compute_phrase_length(pred_bboxes[pred_index]) > 5 else 0.5
            else:
                threshold = th_iou

            if pred_cls[pred_index] in gt_cls:  # ground truth is true
                hit_index = gt_cls.index(pred_cls[pred_index])
                if max([gt_cls.count(item) for item in gt_cls]) > 1 and len(
                        np.where(np.array(gt_cls) == pred_cls[pred_index])[0]) > 1:
                    hit_index = np.argmax(
                        np.array(ious)[pred_index])
                max_iou = max(np.array(ious)[pred_index][np.where(np.array(gt_cls) == pred_cls[pred_index])[0]])
                if max_iou >= threshold:
                    assert pred_cls[pred_index] == gt_cls[hit_index]
                    if gt_cls_hit_ls[hit_index] == 0:
                        confusion_matrix[gt_cls[hit_index] - 1, pred_cls[pred_index] - 1] += 1
                        # pred_cls_hit_ls[pred_index] += 1
                        gt_cls_hit_ls[hit_index] += 1
                else:
                    unhitted_index = np.where(np.array(gt_cls_hit_ls) == 0)[0]
                    other_index = np.where(np.array(gt_cls)[unhitted_index] != pred_cls[pred_index])[0]
                    if len(other_index) == 0:
                        confusion_matrix[classes_num, pred_cls[pred_index] - 1] += 1
                        continue
                    hit_index = np.argmax(np.array(ious)[pred_index][other_index])
                    # hit_index = np.argmax(np.array(ious)[pred_index])
                    max_iou = np.array(ious)[pred_index, hit_index]
                    if max_iou >= threshold:
                        confusion_matrix[gt_cls[hit_index] - 1, pred_cls[pred_index] - 1] += 1
                        pre_cls_hit_ls[hit_index] += 1
                    else:
                        confusion_matrix[classes_num, pred_cls[pred_index] - 1] += 1
                    not_hit[pred_cls[pred_index] - 1].append((pred_bboxes[pred_index],
                                                              gt_bboxes[hit_index]))  # todo
            else:
                unhitted_index = np.where(np.array(gt_cls_hit_ls) == 0)[0]
                other_index = np.where(np.array(gt_cls)[unhitted_index] != pred_cls[pred_index])[0]
                if len(other_index) == 0:
                    confusion_matrix[classes_num, pred_cls[pred_index] - 1] += 1
                    continue
                hit_index = np.argmax(np.array(ious)[pred_index][other_index])
                max_iou = np.array(ious)[pred_index, hit_index]
                if max_iou >= threshold:
                    confusion_matrix[gt_cls[hit_index] - 1, pred_cls[pred_index] - 1] += 1
                    pre_cls_hit_ls[hit_index] += 1
                else:
                    confusion_matrix[classes_num, pred_cls[pred_index] - 1] += 1

    for i in range(len(gt_cls_hit_ls)):
        if gt_cls_hit_ls[i] == 0 and pre_cls_hit_ls[i] == 0:
            confusion_matrix[gt_cls[i] - 1, classes_num] += 1
    return


def compute_phrase_length(np_nadrray_1d):
    return np_nadrray_1d[1] - np_nadrray_1d[0]


def get_count_dataframe_by_confusion_matrix(confusion_matrix):
    experiment_metrics = []
    sum_result_dir = {"TP": 0, "FP": 0, "FN": 0}
    for one_label in range(classes_num):
        TP = confusion_matrix[one_label][one_label]
        sum_result_dir["TP"] += TP
        FP = 0
        for column_index in range(classes_num + 1):
            FP += confusion_matrix[column_index][one_label]
        FP -= TP
        sum_result_dir["FP"] += FP
        FN = sum(confusion_matrix[one_label]) - TP

        sum_result_dir["FN"] += FN
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        F1_score = (2 * precision * recall) / (precision + recall) if precision * recall != 0 else 0
        experiment_metrics.append([precision, recall, F1_score])

    TP = sum_result_dir["TP"]
    FP = sum_result_dir["FP"]
    FN = sum_result_dir["FN"]
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    F1_score = (2 * precision * recall) / (precision + recall) if precision * recall != 0 else 0

    experiment_metrics.append([precision, recall, F1_score])
    return pd.DataFrame(experiment_metrics, columns=["precision", "recall", "F1_score"],
                        index=cfg.LABEL[0: classes_num] + ["overall"])
