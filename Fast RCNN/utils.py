# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2018-10-06

import re
import jieba
import pynlpir
import numpy as np
import pandas as pd
# import gensim
from gensim.models import Word2Vec
import config as cfg

ignore_str = "，。,；：:…． "
split_chars = '[，。,；：:…．;]'

word_segmentation_method = cfg.WORD_SEGMENTATION_METHOD
# if word_segmentation_method:
word_vector_model = Word2Vec.load(cfg.JIEBA_WV_MODEL)
# else:
#     word_vector_model = gensim.models.KeyedVectors.load_word2vec_format(config_genia.BIO_WV_MODEL, binary=True)

jieba.load_userdict(cfg.JIEBA_USER_DICT)

word_embedding_dim = cfg.WORD_EMBEDDING_DIM
sentence_length = cfg.SENTENCE_LENGTH
empty_wv = np.zeros((1, word_embedding_dim))
classes_num = cfg.CLASSES_NUM
# th_iou_train = cfg.TH_IOU_TRAIN


not_hit = [[] for i in range(classes_num)]
hit = [0]
#  0.6
norm_info = {"0.6": {
    "all_mean": [[0.06443895298506289, 0.24921900449936088], [0.07358772181097258, 0.25045570684169505],
                 [0.08552335185463283, 0.25378268665231224], [0.0722001228938262, 0.2499207120587632],
                 [0.0690014425388684, 0.2573712646714699]],
    "all_deviation": [[2.193887953222562, 0.20025750261275815], [2.2718000657373274, 0.19973858438050374],
                      [2.2150875539497994, 0.1965857106123908], [2.175039202244989, 0.19913427903962],
                      [2.2406560778201627, 0.192379409701591]]
},
    "0.8": {
        "all_mean": [[0.03111014869351812, 0.10445007686158266], [0.037572928821470244, 0.10523528550053966],
                     [0.0418747687754347, 0.10599033194468542], [0.03701144887485196, 0.10409750019725891],
                     [0.03273780308235987, 0.10737650507590252]],
        "all_deviation": [[1.1238326697207834, 0.09710832458180578], [1.1705871043184342, 0.09613037703358712],
                          [1.1449200003450926, 0.09535004808422333], [1.1203488005079396, 0.09727440367749295],
                          [1.1500399367630698, 0.0941451946949091]]
    },
    "0.9": {
        "all_mean": [[0.010611817109917463, 0.04085855679433742], [0.012721665381649962, 0.04099686693343283],
                     [0.01305664185694462, 0.04137338143608464], [0.011869635193133048, 0.040320387189508734],
                     [0.010872447626624237, 0.041767302468701605]],
        "all_deviation": [[0.5699429477422212, 0.046797115821464834], [0.5914874606939978, 0.0468368080526009],
                          [0.5812051916821024, 0.04634683705849847], [0.5695373929985009, 0.047248527422787616],
                          [0.5813492141600299, 0.04616946291496999]]
    }
}


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


def denorm(pred_bbox, k_fold_index, th_train_iou):  # todo
    all_deviation = norm_info[str(th_train_iou)]["all_deviation"]
    all_mean = norm_info[str(th_train_iou)]["all_mean"]
    pred_bbox[:, :, 0] = (pred_bbox[:, :, 0] * all_deviation[k_fold_index][0]) + all_mean[k_fold_index][0]
    pred_bbox[:, :, 1] = (pred_bbox[:, :, 1] * all_deviation[k_fold_index][1]) + all_mean[k_fold_index][1]
    return pred_bbox


def bbox_transform_1d(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 1] - ex_rois[:, 0]
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths

    gt_widths = gt_rois[:, 1] - gt_rois[:, 0]
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths

    # targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dx = (gt_ctr_x - ex_ctr_x)
    targets_dw = np.log(gt_widths / ex_widths)

    targets = np.array([targets_dx, targets_dw]).T
    return targets


def calc_ious_1d(ex_rois, gt_rois):
    try:
        ex_area = ex_rois[:, 1] - ex_rois[:, 0]
    except:
        wait = True
    gt_area = gt_rois[:, 1] - gt_rois[:, 0]
    area_sum = ex_area.reshape((-1, 1)) + gt_area.reshape((1, -1))

    lb = np.maximum(ex_rois[:, 0].reshape((-1, 1)), gt_rois[:, 0].reshape((1, -1)))
    rb = np.minimum(ex_rois[:, 1].reshape((-1, 1)), gt_rois[:, 1].reshape((1, -1)))

    area_i = np.maximum(rb - lb, 0)
    area_u = area_sum - area_i
    ious = area_i / area_u
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


def str2wv(ls):
    arr = empty_wv
    for i in ls:
        try:
            arr = np.vstack((arr, word_vector_model.wv[i]))
        except KeyError:
            arr = np.vstack((arr, empty_wv))
            # print(j, "can not be found in word2vec dictionary")
    arr = np.delete(arr, 0, axis=0)
    row, column = arr.shape
    try:
        arr = np.pad(arr, ((0, sentence_length - row), (0, 0)), 'constant', constant_values=0)
    except Exception as e:
        print(list2str(ls))
        exit("Above sentence length is " + str(len(ls)) + " which was too long")
    return arr


def list2str(ls):
    return "".join(ls)


def txt_split(text_str):
    """Input: one paragraph.
        Return: sentence word cut list.
        like: [["w1", "w2", "w3"], ["w1", "w2"]].
        For each one sentence in this paragraph, we return the word segmentation of one sentence.
    """
    ls_seg_sentences = []
    if word_segmentation_method == "jieba":
        for item in [sentence for sentence in re.split(split_chars, text_str) if len(sentence) > 0]:
            ls_seg_sentences.append([seg for seg in jieba.cut(item)])
    elif word_segmentation_method == "nlpir":
        for item in [sentence for sentence in re.split(split_chars, text_str) if len(sentence) > 0]:
            ls_seg_sentences.append([seg for seg, flag in pynlpir.segment(item)])
    else:
        raise KeyError("Word segmentation method was invalid")
    return ls_seg_sentences


def reg_to_bbox(sentence_length, reg, box):  # todo   to be revised.
    bbox_width = box[:, 1] - box[:, 0]  # Region length
    bbox_ctr_x = box[:, 0] + 0.5 * bbox_width  # x coordinate   # Region coordinate

    bbox_width = bbox_width[:, np.newaxis]
    bbox_ctr_x = bbox_ctr_x[:, np.newaxis]

    # out_ctr_x = reg[:, :, 0] * bbox_width + bbox_ctr_x
    out_ctr_x = reg[:, :, 0] + bbox_ctr_x

    out_width = bbox_width * np.exp(reg[:, :, 1])

    return np.array([
        np.maximum(0, out_ctr_x - 0.5 * out_width),
        np.minimum(sentence_length, out_ctr_x + 0.5 * out_width), ])


def non_maximum_suppression(scores, bboxs, iou_threshold=0.5, score_threshold=0.6):  # todo   to be revised.
    nroi = scores.shape[0]
    # sort the roi score and get it's index from big to small
    idx = np.argsort(scores)[::-1]
    right_boundary = 0
    while right_boundary < nroi and scores[idx[right_boundary]] >= score_threshold:
        right_boundary += 1
    if right_boundary == 0:
        return []
    idx = idx[:right_boundary]
    if len(idx) > 2:
        wait = True
    # scores = scores[idx]
    bboxs = bboxs[idx, :]
    ious = calc_ious_1d(bboxs, bboxs)

    res = []
    res_index = []
    for i in range(right_boundary):
        # if i == 0 or ious[i, :i].max() < iou_threshold:
        if i == 0 or ious[i, res_index].max() < iou_threshold:  # rectify in 10-29
            res.append(bboxs[i])
            res_index.append(i)
            if i != 0:
                wait = True

    return res


def evaluate(pred_bboxes, pred_cls, info, confusion_matrix, th_iou=0.6):  # todo
    assert len(pred_bboxes) == len(pred_cls)
    gt_bboxes = info["gt_bboxs"]
    gt_cls = info["gt_cls"]
    gt_str_space = info["gt_str"]
    gt_cls_hit_ls = [0 for i in range(len(gt_cls))]
    s_length = len(gt_str_space.split(" "))
    if pred_bboxes.size > 0 and th_iou == 1:
        pred_bboxes = np.round(pred_bboxes, 0)
        pred_bboxes[:, 0][pred_bboxes[:, 0] < 0] = 0
        pred_bboxes[:, 1][pred_bboxes[:, 1] > s_length] = s_length
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
                if th_iou != 1:
                    threshold = th_iou if compute_phrase_length(pred_bboxes[pred_index]) >= 4 else 0.4
                else:
                    threshold = th_iou
                # for instance 1 1 4 this situation hit index could be duplicated.
                max_iou = max(np.array(ious)[pred_index][np.where(np.array(gt_cls) == pred_cls[pred_index])[0]])
                # if gt_cls.count(pred_cls[pred_index]) > 1:
                #     max_iou = max(np.array(ious)[pred_index][np.where(np.array(gt_cls) == pred_cls[pred_index])[0]])
                # else:
                #     max_iou = ious[pred_index][hit_index]
                # except:
                #     wait=True
                if max_iou >= threshold:
                    assert pred_cls[pred_index] == gt_cls[hit_index]
                    confusion_matrix[pred_cls[pred_index] - 1, pred_cls[pred_index] - 1] += 1
                    # pred_cls_hit_ls[pred_index] += 1
                    gt_cls_hit_ls[hit_index] += 1
                # elif np.where(cls_array[pred_index] == True).size > 0:  # at least hit another ground truth
                #     confusion_matrix[pred_cls[pred_index] - 1, np.where(cls_array[pred_index] == True)] += 1
                else:
                    hit_flag = False
                    if pred_cls[pred_index] == 1 and th_iou != 1:
                        pred_str = "".join(gt_str_space.split(" ")[int(round(pred_bboxes[pred_index][0])): int(
                            round(pred_bboxes[pred_index][1]))])
                        gt_1_index_ls = [index for index in range(len(gt_cls)) if gt_cls[index] == 1]
                        for x in range(len(gt_1_index_ls)):
                            gt_str = "".join(
                                gt_str_space.split(" ")[gt_bboxes[gt_1_index_ls[x]][0]: gt_bboxes[gt_1_index_ls[x]][1]])
                            if pred_str.find(gt_str) > -1 or gt_str.find(pred_str) > -1:
                                gt_cls_hit_ls[gt_1_index_ls[x]] += 1  # hit
                                confusion_matrix[0, 0] += 1
                                hit_flag = True
                                hit[0] += 1
                    if not hit_flag:
                        confusion_matrix[
                            classes_num, pred_cls[pred_index] - 1] += 1  # predict do not hit any ground truth
                        not_hit[pred_cls[pred_index] - 1].append((pred_bboxes[pred_index],
                                                                  gt_bboxes[hit_index]))  # todo
                    if max_iou != 0:
                        wait = True
            else:
                confusion_matrix[classes_num, pred_cls[pred_index] - 1] += 1

    for i in range(len(gt_cls_hit_ls)):
        if gt_cls_hit_ls[i] == 0:
            confusion_matrix[gt_cls[i] - 1, classes_num] += 1
            if gt_cls[i] == 1:
                wait = True
    return


def gt_evaluate(pred_bboxes, pred_cls, info, confusion_matrix, th_iou=0.6):  # todo
    assert len(pred_bboxes) == len(pred_cls)
    # np.where(pred_cls!=0)
    gt_bboxes = info["gt_bboxs"]
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
                if max_iou > threshold:
                    if gt_cls_hit_ls[hit_index] != 0:
                        continue
                    assert pred_cls[pred_index] == gt_cls[hit_index]
                    confusion_matrix[pred_cls[pred_index] - 1, pred_cls[pred_index] - 1] += 1
                    # pred_cls_hit_ls[pred_index] += 1
                    gt_cls_hit_ls[hit_index] += 1
                else:
                    hit_flag = False
                    if pred_cls[pred_index] == 1:
                        pred_str = "".join(gt_str_space.split(" ")[int(round(pred_bboxes[pred_index][0])): int(
                            round(pred_bboxes[pred_index][1]))])
                        gt_1_index_ls = [index for index in range(len(gt_cls)) if gt_cls[index] == 1]
                        for x in range(len(gt_1_index_ls)):
                            gt_str = "".join(
                                gt_str_space.split(" ")[gt_bboxes[gt_1_index_ls[x]][0]: gt_bboxes[gt_1_index_ls[x]][1]])
                            if pred_str.find(gt_str) > -1 or gt_str.find(pred_str) > -1:
                                gt_cls_hit_ls[gt_1_index_ls[x]] += 1  # hit
                                confusion_matrix[0, 0] += 1
                                hit_flag = True
                                hit[0] += 1
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
    return pd.DataFrame(experiment_metrics, columns=["precision", "recall", "F1_score"], index=cfg.LABEL + ["overall"])
