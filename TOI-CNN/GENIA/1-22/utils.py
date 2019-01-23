# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2018-10-06

import re
import jieba
import numpy as np
import pandas as pd
# import gensim
import config as cfg
import queue

ignore_str = "，。,；：:…． "
split_chars = '[，。,；：:…．;]'

# word_segmentation_method = cfg.WORD_SEGMENTATION_METHOD
# jieba.load_userdict(cfg.JIEBA_USER_DICT)

word_embedding_dim = cfg.WORD_EMBEDDING_DIM
sentence_length = cfg.SENTENCE_LENGTH
empty_wv = np.zeros((1, word_embedding_dim))
classes_num = cfg.CLASSES_NUM
# th_iou_train = cfg.TH_IOU_TRAIN

# contain_matrix = [[0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0],
#                   [599, 0, 113, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0]]

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
        "left_boundary": {
            "all_mean": [[-2.2831599326334344, 0.24921900449936088], [-2.3766376697975247, 0.25045570684169505],
                         [-2.3828652950893527, 0.25378268665231224], [-2.243151903237282, 0.2499207120587632],
                         [-2.3880108991825613, 0.2573712646714699]],
            "all_deviation": [[3.1510883343378837, 0.20025750261275815], [3.243162736750187, 0.19973858438050374],
                              [3.1915002819666904, 0.1965857106123908], [3.0769540883131996, 0.19913427903962],
                              [3.1638410325460593, 0.192379409701591]]
        }
    },
    "0.7": {
        "centre": {
            "all_mean": [[0.0519726567500448, 0.1677972380286376], [0.05692356891589383, 0.16952092867851598],
                         [0.07198562487776257, 0.17134793884788824], [0.056240226869152685, 0.1680522175879879],
                         [0.05139402523146327, 0.1741314017627747]],
            "all_deviation": [[1.674601725572448, 0.14683484118460496], [1.734678342253432, 0.14636284716428793],
                              [1.6905512941506244, 0.14405277503896227], [1.6576582449664703, 0.14611708779018967],
                              [1.711903266722968, 0.14107311598467517]]
        }
    },
    "0.8": {
        "centre": {
            "all_mean": [[0.00042789223454833597, -0.02895624176034832],
                         [-0.0017078772193660453, -0.02525573581907752]],
            "all_deviation": [[0.4650596693405339, 0.1276511062431574], [0.37139933239690665, 0.12721136448455436]]
        },
        "left_boundary": {
            # to be filled
        }
    },
    "0.9": {
        "centre": {
            "all_mean": [[0.010611817109917463, 0.04085855679433742], [0.012721665381649962, 0.04099686693343283],
                         [0.01305664185694462, 0.04137338143608464], [0.011869635193133048, 0.040320387189508734],
                         [0.010872447626624237, 0.041767302468701605]],
            "all_deviation": [[0.5699429477422212, 0.046797115821464834], [0.5914874606939978, 0.0468368080526009],
                              [0.5812051916821024, 0.04634683705849847], [0.5695373929985009, 0.047248527422787616],
                              [0.5813492141600299, 0.04616946291496999]]
        },
        "left_boundary": {
            # to be filled
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


def denorm(pred_bbox, k_fold_index, test_arguments):
    all_deviation = norm_info[str(test_arguments.th_train_iou)][test_arguments.dx_compute_method]["all_deviation"]
    all_mean = norm_info[str(test_arguments.th_train_iou)][test_arguments.dx_compute_method]["all_mean"]
    pred_bbox[:, :, 0] = (pred_bbox[:, :, 0] * all_deviation[k_fold_index][0]) + all_mean[k_fold_index][0]
    pred_bbox[:, :, 1] = (pred_bbox[:, :, 1] * all_deviation[k_fold_index][1]) + all_mean[k_fold_index][1]
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


# def calc_ious_1d_1(ex_rois, gt_rois):  # modify  in 11-16
#     ex_area = ex_rois[:, 1] - ex_rois[:, 0]
#     gt_area = gt_rois[:, 1] - gt_rois[:, 0]
#
#     area_sum = ex_area.reshape((-1, 1)) + gt_area.reshape((1, -1))
#
#     lb = np.maximum(ex_rois[:, 0].reshape((-1, 1)), gt_rois[:, 0].reshape((1, -1)))
#     rb = np.minimum(ex_rois[:, 1].reshape((-1, 1)), gt_rois[:, 1].reshape((1, -1)))
#
#     area_i = np.maximum(rb - lb, 0)
#     area_u = area_sum - area_i
#     ious = area_i / area_u
#     return ious


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


def str2wv(ls, word_vector_model):  # modify in Arguments
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


def non_maximum_suppression_mix(scores, bboxs, original_roi=[], iou_threshold=0.5, score_threshold=0.6, info=None,
                                return_all_flag=True):
    roi_num = scores.shape[0]
    sentence_length = len(info['gt_str'])
    bboxs_length = bboxs[:, 1] - bboxs[:, 0] + 1
    SLR = bboxs_length / sentence_length
    sorted_lenth_indexs = np.argsort(bboxs_length)[::-1]
    sorted_score_indexs = np.argsort(scores)[::-1]

    lenth_rank = np.zeros(roi_num)
    score_rank = np.zeros(roi_num)
    for i in range(roi_num):
        lenth_rank[sorted_lenth_indexs[i]] = i
        score_rank[sorted_score_indexs[i]] = i
    mix_rank = [lenth_rank[i] * SLR[i] + score_rank[i] * (1 - SLR[i]) for i in range(roi_num)]
    sorted_mix_indexs = np.argsort(mix_rank)

    bboxs = bboxs[sorted_mix_indexs, :]
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
        return result, drop, original_roi[sorted_score_indexs[result_index]], scores[sorted_score_indexs[result_index]]
    else:
        return result


def non_maximum_suppression_l(scores, bboxs, original_roi, iou_threshold=0.5, score_threshold=0.6, info=None,
                              return_all_flag=True):
    roi_num = scores.shape[0]
    sorted_score_indexs = np.argsort(bboxs[:, 1] - bboxs[:, 0])[::-1]  # sort by length.
    bboxs = bboxs[sorted_score_indexs, :]
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
        return result, drop, original_roi[sorted_score_indexs[result_index]], scores[sorted_score_indexs[result_index]]
    else:
        return result


def non_maximum_suppression(scores, bboxs, original_roi, iou_threshold=0.5, score_threshold=0.6, info=None,
                            return_all_flag=True):
    roi_num = scores.shape[0]
    sorted_score_indexs = np.argsort(scores)[::-1]
    bboxs = bboxs[sorted_score_indexs, :]
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
        return result, drop, original_roi[sorted_score_indexs[result_index]], scores[sorted_score_indexs[result_index]]
    else:
        return result


def non_maximum_suppression_all_regression(scores, bboxs, original_roi, iou_threshold=0.5, score_threshold=0.6,
                                           info=None, return_all_flag=True):
    roi_num = scores.shape[0]
    # sort the roi score and get it's index from big to small
    sorted_score_indexs = np.argsort(scores)[::-1]
    score_right_boundary = 0
    while score_right_boundary < roi_num and scores[sorted_score_indexs[score_right_boundary]] >= score_threshold:
        score_right_boundary += 1
    if score_right_boundary == 0:
        return [], [], [], [] if return_all_flag else []
    sorted_score_indexs = sorted_score_indexs[:score_right_boundary]
    if len(sorted_score_indexs) > 2:
        wait = True
    bboxs = bboxs[sorted_score_indexs, :]
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
        return result, drop, original_roi[sorted_score_indexs[result_index]], scores[sorted_score_indexs[result_index]]
    else:
        return result


def rePredic(q):
    """广度优先去除不正确的contain关系
        input: 存储元素的数据结构为[pred_bboxes, pred_cls, score]的队列q

        output: 处理完毕的[pred_bboxes, pred_cls, score]"""
    pred_bboxes, pred_cls, score = q.get()
    len_bbox = len(pred_bboxes)
    if len_bbox == 1 or len_bbox == 0:  # 只剩下0、1个候选框
        return pred_bboxes, pred_cls, score
    isTrue = True  # TOI序列是否正确的标志
    for i in range(len(pred_bboxes)):
        for j in range(i + 1, len(pred_bboxes)):
            if isContain(pred_bboxes[i], pred_bboxes[j]):
                if contain_matrix[pred_cls[i] - 1][pred_cls[j] - 1] == 0:  # 预测结果i contain j但是contain矩阵不允许
                    isTrue = False
                    break
            elif isContain(pred_bboxes[j], pred_bboxes[i]):
                if contain_matrix[pred_cls[j] - 1][pred_cls[i] - 1] == 0:  # 预测结果j contain i但是contain矩阵不允许
                    isTrue = False
                    break
            elif isoverlap(pred_bboxes[i], pred_bboxes[j]):  # 预测结果i和j重叠
                isTrue = False
                break

        if not isTrue:  # 如果内循环出错，退出外循环
            break

    if isTrue:  # 如果TOI序列contain关系正确，返回结果
        return pred_bboxes, pred_cls, score
    # 如果有错误，按照得分从小到大生成新的去除方案放入队列q中
    sorted_index = np.argsort(score)
    pred_bboxes = pred_bboxes[sorted_index, :]
    pred_cls = pred_cls[sorted_index]
    score = score[sorted_index]
    for i in range(len_bbox):
        index = np.hstack([np.arange(0, i), np.arange(i + 1, len_bbox)])  # 去除 i-th TOI
        q.put([pred_bboxes[index], pred_cls[index], score[index]])
    return rePredic(q)


def isContain(bbox1, bbox2):
    '''if bbox1 contain bbox2, return True'''
    x1, y1 = bbox1
    x2, y2 = bbox2
    x = min(x1, x2)
    y = max(y1, y2)
    return x == x1 and y == y1


def isoverlap(bbox1, bbox2):
    x1, y1 = bbox1
    x2, y2 = bbox2
    x = max(x1, x2)
    y = min(y1, y2)
    return x <= y


# 
# def evaluate_in(gold_entities, pred_entities):
#     prec_all_num, prec_num, recall_all_num, recall_num = 0, 0, 0, 0
#     for g_ets, p_ets in zip(gold_entities, pred_entities):
#         recall_all_num += len(g_ets)
#         prec_all_num += len(p_ets)
# 
#         for et in g_ets:
#             if et in p_ets:
#                 recall_num += 1
# 
#         for et in p_ets:
#             if et in g_ets:
#                 prec_num += 1
# 
#     return prec_all_num, prec_num, recall_all_num, recall_num


def evaluate(original_rois, pred_bboxes, pred_cls, drop_bbox, drop_cls, info, confusion_matrix, th_iou=0.6):  # todo
    assert len(pred_bboxes) == len(pred_cls)
    assert len(drop_bbox) == len(drop_cls)
    gt_bboxes = info["gt_bboxs"]
    gt_cls = info["gt_cls"]
    gt_str_space = info["gt_str"]
    gt_cls_hit_ls = [0 for i in range(len(gt_cls))]
    s_length = len(gt_str_space)
    # if pred_bboxes.size > 0 and th_iou == 1:

    TP = 0;
    FP = 0;
    FN = 0
    sub_confusion_matrix = np.zeros((cfg.CLASSES_NUM + 1, cfg.CLASSES_NUM + 1))

    if len(pred_cls) > 0:
        # if len(gt_cls) > 1:
        #     wait = True
        # pred_cls_hit_ls = [0 for i in range(len(pred_cls))]
        # calculate predict and ground truth ious and sort
        ious = calc_ious_1d(pred_bboxes, gt_bboxes)

        # evaluate the pred_clses and gt_clses same

        # pred_cls_extend = np.repeat(pred_cls, pred_cls.shape[1], axis=0).T
        # gt_cls_extend = np.repeat(gt_cls, pred_cls.shape[0], axis=0)
        # cls_evaluation_array = pred_cls_extend == gt_cls_extend

        # numpy ndarray to list!
        for pred_index in range(len(pred_cls)):
            if pred_cls[pred_index] in gt_cls:  # ground truth is true
                hit_index = gt_cls.index(pred_cls[pred_index])
                if max([gt_cls.count(item) for item in gt_cls]) > 1 and len(
                        np.where(np.array(gt_cls) == pred_cls[pred_index])[0]) > 1:
                    hit_index = np.argmax(
                        np.array(ious)[pred_index])
                    wait = True
                if th_iou != 1:
                    threshold = th_iou if compute_phrase_length(pred_bboxes[pred_index]) > 5 else 0.5
                else:
                    threshold = th_iou
                # threshold = th_iou
                # for instance 1 1 4 this situation hit index could be duplicated.
                max_iou = max(np.array(ious)[pred_index][np.where(np.array(gt_cls) == pred_cls[pred_index])[0]])
                if max_iou >= threshold:
                    try:
                        assert pred_cls[pred_index] == gt_cls[hit_index]
                    except:
                        wait = True
                    if gt_cls_hit_ls[hit_index] == 0:
                        confusion_matrix[pred_cls[pred_index] - 1, pred_cls[pred_index] - 1] += 1
                        sub_confusion_matrix[pred_cls[pred_index] - 1, pred_cls[pred_index] - 1] += 1
                        TP += 1
                        # pred_cls_hit_ls[pred_index] += 1
                        gt_cls_hit_ls[hit_index] += 1
                else:
                    # relax criterion
                    # hit_flag = False
                    # if pred_cls[pred_index] == 1 and th_iou != 1:
                    #     pred_str = "".join(gt_str_space[int(round(pred_bboxes[pred_index][0])): int(
                    #         round(pred_bboxes[pred_index][1]))])
                    #     gt_1_index_ls = [index for index in range(len(gt_cls)) if gt_cls[index] == 1]
                    #     for x in range(len(gt_1_index_ls)):
                    #         gt_str = "".join(
                    #             gt_str_space[gt_bboxes[gt_1_index_ls[x]][0]: gt_bboxes[gt_1_index_ls[x]][1]])
                    #         if pred_str.find(gt_str) > -1 or gt_str.find(pred_str) > -1:
                    #             gt_cls_hit_ls[gt_1_index_ls[x]] += 1  # hit
                    #             confusion_matrix[0, 0] += 1
                    #             hit_flag = True
                    #             hit[0] += 1
                    # if not hit_flag:
                    confusion_matrix[
                        classes_num, pred_cls[pred_index] - 1] += 1  # predict do not hit any ground truth
                    sub_confusion_matrix[classes_num, pred_cls[pred_index] - 1] += 1
                    FP += 1
                    not_hit[pred_cls[pred_index] - 1].append((pred_bboxes[pred_index],
                                                              gt_bboxes[hit_index]))  # todo
                    # if pred_cls[pred_index] == 4:
                    #     print("原句：", "".join(gt_str_space))
                    #     print("FP：",
                    #           "".join(gt_str_space[int(pred_bboxes[pred_index][0]):int(pred_bboxes[pred_index][1]) + 1]))
                    #     # print(pred_bboxes[pred_index][0])
                    #     print("Class Type:", cfg.LABEL[pred_cls[pred_index] - 1])
                    #     print("错误类型：", "th_p iou不足\n\n")
            else:
                confusion_matrix[classes_num, pred_cls[pred_index] - 1] += 1
                sub_confusion_matrix[classes_num, pred_cls[pred_index] - 1] += 1
                FP += 1
                # if pred_cls[pred_index] == 4:
                #     print("原句：", "".join(gt_str_space))
                #     print("FP：",
                #           "".join(gt_str_space[int(pred_bboxes[pred_index][0]):int(pred_bboxes[pred_index][1]) + 1]))
                #     print("Class Type:", cfg.LABEL[pred_cls[pred_index] - 1])
                #     print("错误类型：", "不在原句中\n\n")

    # check the drop bbox if hit:
    # if min(gt_cls_hit_ls) != 0 and len(drop_cls) > 0:  # lose hit at least one
    #     drop_ious = calc_ious_1d(drop_bbox, gt_bboxes)
    #     for drop_index in range(len(drop_cls)):
    #         if drop_cls[drop_index] in gt_cls:  # ground truth is true
    #             hit_index = gt_cls.index(drop_cls[drop_index])
    #             if th_iou != 1:
    #                 threshold = th_iou if compute_phrase_length(drop_bbox[drop_index]) >= 4 else 0.4
    #             else:
    #                 threshold = th_iou
    #             if max([gt_cls.count(item) for item in gt_cls]) > 1 and len(
    #                     np.where(np.array(gt_cls) == drop_cls[drop_index])[0]) > 1:
    #                 hit_index = np.argmax(
    #                     np.array(drop_ious)[drop_index])
    #             # for instance 1 1 4 this situation hit index could be duplicated.
    #             max_iou = max(np.array(drop_ious)[drop_index][np.where(np.array(gt_cls) == drop_cls[drop_index])[0]])
    #             # if max_iou>= threshold:
    #             #
    #             if max_iou >= threshold and gt_cls_hit_ls[hit_index] == 0:
    #                 nms_mis_hit[drop_cls[drop_index] - 1] += 1

    for i in range(len(gt_cls_hit_ls)):
        if gt_cls_hit_ls[i] == 0:
            confusion_matrix[gt_cls[i] - 1, classes_num] += 1
            sub_confusion_matrix[gt_cls[i] - 1, classes_num] += 1
            FN += 1
            # print("原句：", "".join(gt_str_space))
            # print("FN：", "".join(gt_str_space[gt_bboxes[i][0]: gt_bboxes[i][1] + 1]))

            if gt_cls[i] not in list(drop_cls) + list(pred_cls):
                not_detect_ls[gt_cls[i] - 1] += 1
                # print("Not even classify right")
            elif gt_cls[i] not in list(pred_cls) and gt_cls[i] in list(drop_cls):
                nms_mis_hit[gt_cls[i] - 1] += 1
                # print("NMS Lose")
            elif gt_cls[i] in list(pred_cls):
                iou_lower_hit[gt_cls[i] - 1] += 1
                # print("IOU lower than threshold")
            # print("Label：", cfg.LABEL[gt_cls[i] - 1])
            # print("\n\n")
            if gt_cls[i] == 3:
                wait = True

    # outfile = open('./evaluate1.data', 'a')
    # outfile.write(' '.join(info['gt_str']) + '\ngt: ')
    # for i in range(len(info['gt_cls'])):
    #     outfile.write(cfg.LABEL[info['gt_cls'][i] - 1] + str(info['gt_bboxs'][i]) + '\t')
    # outfile.write('\npre: ')
    # for i in range(len(pred_bboxes)):
    #     outfile.write(cfg.LABEL[pred_cls[i] - 1] + str(pred_bboxes[i]).strip('.') + '\t')
    # outfile.write('\n')
    # outfile.write(f'TP:{TP}\tFP:{FP}\tFN:{FN}\n')
    # #outfile.write(str(sub_confusion_matrix) + '\n\n')
    # outfile.close()

    return


def my_evaluate(pred_bboxes, pred_cls, info, confusion_matrix, th_iou=0.6):
    assert len(pred_bboxes) == len(pred_cls)
    gt_bboxes = info["gt_bboxs"]
    gt_cls = info["gt_cls"]
    gt_str_space = info["gt_str"]
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
                    try:
                        assert pred_cls[pred_index] == gt_cls[hit_index]
                    except:
                        wait = True
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


def gt_evaluate(pred_bboxes, pred_cls, info, confusion_matrix, th_iou=0.6):  # todo
    assert len(pred_bboxes) == len(pred_cls)
    # np.where(pred_cls!=0)
    gt_bboxes = np.copy(info["gt_bboxs"])
    gt_cls = info["gt_cls"]
    gt_str_space = info["gt_str"]
    gt_cls_hit_ls = [0 for i in range(len(gt_cls))]
    if len(pred_cls) > 0:
        ious = calc_ious_1d(pred_bboxes, gt_bboxes)
        for pred_index in range(len(pred_cls)):
            if pred_cls[pred_index] in gt_cls and pred_cls[pred_index] > 0:  # ground truth is true
                hit_index = gt_cls.index(pred_cls[pred_index])
                threshold = th_iou if compute_phrase_length(pred_bboxes[pred_index]) >= 4 else 0.4
                max_iou = max(np.array(ious)[pred_index][np.where(np.array(gt_cls) == pred_cls[pred_index])[0]])
                if max_iou >= threshold:
                    if gt_cls_hit_ls[hit_index] != 0:
                        continue
                    assert pred_cls[pred_index] == gt_cls[hit_index]
                    confusion_matrix[pred_cls[pred_index] - 1, pred_cls[pred_index] - 1] += 1
                    # pred_cls_hit_ls[pred_index] += 1
                    gt_cls_hit_ls[hit_index] += 1
                else:
                    hit_flag = False
                    # if pred_cls[pred_index] == 1:
                    #     # pred_str = "".join(gt_str_space.split(" ")[int(round(pred_bboxes[pred_index][0])): int(
                    #     #     round(pred_bboxes[pred_index][1]))])
                    #     pred_str = "".join(gt_str_space[int(round(pred_bboxes[pred_index][0])): int(
                    #         round(pred_bboxes[pred_index][1]))])
                    #     gt_1_index_ls = [index for index in range(len(gt_cls)) if gt_cls[index] == 1]
                    #     for x in range(len(gt_1_index_ls)):
                    #         # gt_str = "".join(
                    #         #     gt_str_space.split(" ")[gt_bboxes[gt_1_index_ls[x]][0]: gt_bboxes[gt_1_index_ls[x]][1]])
                    #         gt_str = "".join(
                    #             gt_str_space[gt_bboxes[gt_1_index_ls[x]][0]: gt_bboxes[gt_1_index_ls[x]][1]])
                    #         if pred_str.find(gt_str) > -1 or gt_str.find(pred_str) > -1:
                    #             gt_cls_hit_ls[gt_1_index_ls[x]] += 1  # hit
                    #             confusion_matrix[0, 0] += 1
                    #             hit_flag = True
                    #             hit[0] += 1
                    if not hit_flag:
                        confusion_matrix[
                            classes_num, pred_cls[pred_index] - 1] += 1  # predict do not hit any ground truth
                        not_hit[pred_cls[pred_index] - 1].append((pred_bboxes[pred_index],
                                                                  gt_bboxes[hit_index]))  # todo
            else:
                if pred_cls[pred_index] != 0:
                    confusion_matrix[classes_num, pred_cls[pred_index] - 1] += 1

    for i in range(len(gt_cls_hit_ls)):
        if gt_cls_hit_ls[i] == 0:
            confusion_matrix[gt_cls[i] - 1, classes_num] += 1
            if gt_cls[i] == 1:
                wait = True
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
