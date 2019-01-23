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
from utils import reg_to_bbox, non_maximum_suppression, evaluate, get_count_dataframe_by_confusion_matrix, denorm
from utils import first_write, lb_reg_to_bbox
from utils import non_maximum_suppression_all_regression, non_maximum_suppression_l, non_maximum_suppression_mix
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


def check_regression(pred_bbox, rois, original_pred_cls_score, sentence_len, th_train_iou):
    final_pred_bbox = int_bbox(pred_bbox, sentence_len)
    ious = utils.calc_ious_1d(final_pred_bbox, rois)
    index = np.array([True if ious[i, i] >= th_train_iou else False for i in range(len(pred_bbox))])   # 回归范围超出训练阈值的候选框
    original_pred_cls_score[index, 1:cfg.CLASSES_NUM + 1] -= 0.0    # 非bg类型得分调整
    try:
        a = pred_bbox[index], rois[index], F.softmax(t.Tensor(original_pred_cls_score[index]), dim=1).data.cpu().numpy()
    except:
        return pred_bbox[index], rois[index], np.array([])
    return a


def _test_one_sentence(file_before, test_arguments, sentence, rois, rcnn, fold_index, this_sentence_len,
                       info):
    roi_num = rois.shape[0]
    ridx = np.zeros(roi_num).astype(int)
    if roi_num != 0:    # 候选框数量不为0
        original_pred_cls_score, pred_tbbox = rcnn(sentence, rois, ridx)
        pred_cls_score = F.softmax(original_pred_cls_score, dim=1).data.cpu().numpy()  # softmax score for each row
        argmax_softmax = np.argmax(pred_cls_score, axis=1)  # 预测类型
        if test_arguments.loss_weight_lambda == 0:
            pred_bbox = rois
            write_result(pred_bbox, argmax_softmax, np.max(pred_cls_score, axis=1),
                         test_arguments.result_output, info['gt_str'], None, None, 'TOI')   # 输出TOI
        else:   # LR模型需要对候选框进行回归
            pred_tbbox = pred_tbbox.data.cpu().numpy()
            if test_arguments.normalize:
                pred_tbbox = denorm(pred_tbbox, fold_index, test_arguments)
            if test_arguments.dx_compute_method == "left_boundary":
                pred_bbox = lb_reg_to_bbox(this_sentence_len, pred_tbbox, rois)
            else:
                pred_bbox = reg_to_bbox(this_sentence_len, pred_tbbox, rois)

            new_pred_bbox = np.zeros(
                (pred_bbox.shape[1], pred_bbox.shape[0]))  # only select the meaningful bbox regression.
            for i in range(pred_bbox.shape[1]):  # for n region
                new_pred_bbox[i, :] = pred_bbox[:, i, argmax_softmax[i]]
            write_result(int_bbox(new_pred_bbox, this_sentence_len).astype(np.int32), argmax_softmax,
                         np.max(pred_cls_score, axis=1),
                         test_arguments.result_output, info['gt_str'], None, None, 'TOI')   # 输出TOI

            # 去除回归过大的候选框，输出结果
            pred_bbox, rois, pred_cls_score = check_regression(new_pred_bbox, rois,
                                                               original_pred_cls_score.data.cpu().numpy(),
                                                               this_sentence_len, test_arguments.th_train_iou)
            write_result(pred_bbox.astype(np.int32), np.argmax(pred_cls_score, axis=1), np.max(pred_cls_score, axis=1),
                         test_arguments.result_output, info['gt_str'], None, None, 'after check LR')
    else:
        pred_bbox = np.array([])

    result_bbox = []
    result_cls = []
    drop_bbox = []
    drop_cls = []
    original_roi_ls = []
    c_score_ls = []

    if len(pred_bbox) > 0:
        # if True:
        argmax_softmax = np.argmax(pred_cls_score, axis=1)  # 各种处理过后，重新判断预测类型
        # pred_bbox, argmax_softmax = select_meaningful_bbox_regression(pred_bbox, pred_cls_score)

        # filter去除得分小于s_th的候选框，输出结果
        score_indexs = score_filter(pred_cls_score, score_th=test_arguments.score_threshold)
        pred_bbox = pred_bbox[score_indexs, :]
        argmax_softmax = argmax_softmax[score_indexs]
        pred_cls_score = pred_cls_score[score_indexs]
        rois = rois[score_indexs]
        write_result(pred_bbox.astype(np.int32), argmax_softmax, np.max(pred_cls_score, axis=1),
                     test_arguments.result_output, info['gt_str'], None, None, 'after filter')

        # output_bbox = pred_bbox.T.copy()
        # output_result_cls = np.argmax(pred_cls_score, axis=1).copy()
        # int_bbox(output_bbox, len(info["gt_str"]))
        # output_file_result(file_before, output_bbox, output_result_cls, rois, info, np.max(pred_cls_score, axis=1))
        # if test_arguments.output_flow:
        #     output_detect_result(output_bbox, output_result_cls, rois, info, np.max(pred_cls_score, axis=1))
        # 
        # if test_arguments.output_flow:
        #     output_detect_result(result_bbox, result_cls, original_rois, info, scores)

        # pred bbox shape is 2 * n_roi
        # 按照类型进行nms： mix、length、score
        for c in range(1, classes_num + 1):
            cls_index = np.where(argmax_softmax == c)
            if len(cls_index[0]) == 0:  # this cls type is empty
                continue
            c_cls_score = pred_cls_score[cls_index[0], c]
            c_bboxs = pred_bbox[cls_index[0], :]
            original_roi = rois[np.where(argmax_softmax == c)[0], :]
            if c == 1 or c == 7:
                boxes, _boxes, roi, c_score = non_maximum_suppression_mix(c_cls_score, c_bboxs, original_roi,
                                                                      iou_threshold=test_arguments.th_nms_iou,
                                                                      score_threshold=test_arguments.score_threshold,
                                                                      info=info)
            else:
                boxes, _boxes, roi, c_score = non_maximum_suppression_mix(c_cls_score, c_bboxs, original_roi,
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


def _test_epoch(file_before, file_after, test_arguments, fold_index):
    rcnn = load_model(test_arguments)
    test_sentences, test_sentence_info, test_roi = load_test_sentence(test_arguments.test_sentence_npz_path)

    sentence_num = test_sentences.size(0)

    for index in range(sentence_num):
        sentence = Variable(test_sentences[index:index + 1]).cuda()

        info = test_sentence_info[index]

        idxs = info['roi_ids']
        this_sentence_len = len(info['gt_str']) - 1
        rois = test_roi[idxs] if len(idxs) != 0 else np.array([])

        # 输出测试样句信息
        write_result(None, None, None, test_arguments.result_output, info['gt_str'], info['gt_bboxs'],
                     info['gt_cls'], 'begin')

        result_bbox, result_cls, drop_bbox, drop_cls, original_rois, scores = _test_one_sentence(
            file_before,
            test_arguments,
            sentence, rois,
            rcnn,
            fold_index,
            this_sentence_len,
            info=info)
        # todo modify the sentence result only the max softmax result.
        if test_arguments.loss_weight_lambda != 0:
            result_bbox = int_bbox(result_bbox, this_sentence_len)
            drop_bbox = int_bbox(drop_bbox, this_sentence_len)

        # 输出nms之后的结果
        write_result(result_bbox.astype(np.int32), result_cls, scores, test_arguments.result_output, info['gt_str'],
                     None, None, 'after NMS')

        if test_arguments.output_flow:
            output_detect_result(result_bbox, result_cls, original_rois, info, scores)
        output_file_result(file_after, result_bbox, result_cls, original_rois, info, scores)

        # 广度优先去除错误的contain关系，输出结果
        try:
            q = queue.Queue()
            q.put([result_bbox, result_cls, scores])
            pred_bboxes, pred_cls, score = utils.rePredic(q)
        except:
            pred_bboxes, pred_cls, score = result_bbox, result_cls, scores
        write_result(result_bbox.astype(np.int32), result_cls, scores, test_arguments.result_output, info['gt_str'],
                     None, None, 'after delete contain')

        # 更新pre_contain关系矩阵
        for i in range(len(pred_bboxes)):
            for j in range(i + 1, len(pred_bboxes)):
                if isContain(pred_bboxes[i], pred_bboxes[j]):
                    test_arguments.contain_matrix[pred_cls[i] - 1, pred_cls[j] - 1] += 1
                    if isNeedCheck(pred_cls[i], pred_cls[j]):
                        wait = True
                elif isContain(pred_bboxes[j], pred_bboxes[i]):
                    test_arguments.contain_matrix[pred_cls[j] - 1, pred_cls[i] - 1] += 1
                    if isNeedCheck(pred_cls[j], pred_cls[i]):
                        wait = True
                elif isoverlap(pred_bboxes[i], pred_bboxes[j]):
                    test_arguments.overlap_matrix[pred_cls[i] - 1, pred_cls[j] - 1] += 1
        # 更新gt_contain关系矩阵
        # gt_bboxes = info["gt_bboxs"]
        # gt_cls = info["gt_cls"]
        # for i in range(len(gt_bboxes)):
        #     for j in range(i + 1, len(gt_bboxes)):
        #         if isContain(gt_bboxes[i], gt_bboxes[j]):
        #             test_arguments.gt_contain_matrix[gt_cls[i] - 1, gt_cls[j] - 1] += 1
        #             # if isNeedCheck(gt_cls[i],gt_cls[j]):
        #             #     print(f'fold:{fold_index + 1},row:{index * 3 + 1}')
        #         elif isContain(gt_bboxes[j], gt_bboxes[i]):
        #             test_arguments.gt_contain_matrix[gt_cls[j] - 1, gt_cls[i] - 1] += 1
        #             # if isNeedCheck(gt_cls[j],gt_cls[i]):
        #             #     print(f'fold:{fold_index + 1},row:{index * 3 + 1}')
        #         elif isoverlap(gt_bboxes[i], gt_bboxes[j]):
        #             test_arguments.gt_overlap_matrix[gt_cls[i] - 1, gt_cls[j] - 1] += 1

        # 更新混淆矩阵
        utils.my_evaluate(pred_bboxes, pred_cls, info, test_arguments.confusion_matrix, test_arguments.th_iou_p)
        # evaluate(original_rois, pred_bboxes, pred_cls, drop_bbox, drop_cls, info, test_arguments.confusion_matrix,
        #          test_arguments.th_iou_p)

    return



def write_result(pred_bboxes, pred_cls, score, outfile, gt_str, gt_bboxes, gt_cls, type):
    if type == 'begin':
        outfile.write(' '.join(gt_str) + '\ngt: ')
        for i in range(len(gt_cls)):
            outfile.write(
                cfg.LABEL[gt_cls[i] - 1] + ' [' + ' '.join(gt_str[gt_bboxes[i][0]: gt_bboxes[i][1] + 1]) + ']  ')
    else:
        outfile.write('--------------------------------------------------------\n' + type + '\n')
        for i in range(len(pred_cls)):
            if pred_cls[i] != 0:
                outfile.write(
                    cfg.LABEL[pred_cls[i] - 1] + ' [' + ' '.join(gt_str[pred_bboxes[i][0]: pred_bboxes[i][1] + 1])
                    + '] ' + str(pred_bboxes[i]) + str(np.round(score[i], 3)) + '\t')
    outfile.write('\n')
    if type == 'after delete contain':
        outfile.write('\n_______________________________________\n')
    return


def isNeedCheck(x, y):
    if x == y:
        return True
    if y == 2 or y == 4 or y == 5 or y == 7 or y == 6:
        return True
    if x == 5:
        return True
    return False


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
        print(test_arguments.confusion_matrix.astype(np.int32))

        # 输出结果（孙哥说混淆矩阵可以不用了，直接用TP、FP记录和计算指标）
        test_arguments.confusion_matrix = test_arguments.confusion_matrix.astype(np.int32).tolist()
        test_arguments.data_frame = get_count_dataframe_by_confusion_matrix(test_arguments.confusion_matrix)
        # print(utils.nms_mis_hit)
        # utils.nms_mis_hit = [0] * 7
        test_arguments.write_one_result(all_csv_result)
        print(test_arguments.data_frame)
        # print('overlap:')
        # print(str(test_arguments.overlap_matrix))

        # contain矩阵（孙哥说改成预测的正确的contain的矩阵，现在只是预测结果的contain关系）
        print('contain:')
        print(str(test_arguments.contain_matrix))
        # print('gt_overlap:')
        # print(str(test_arguments.gt_overlap_matrix))
        # print('gt_contain:')
        # print(str(test_arguments.gt_contain_matrix))
    return


def main():
    pos_loss_method = "mse"  # "mse"
    th_train_iou = 0.8  # 0.8
    norm = True  # False
    min_test_epoch = 31
    max_test_epoch = 31
    loss_weight_lambdas = [0]  # [0.1, 0.2, 0.5, 1, 2, 5, 10]
    prevent_overfitting_method = "Dropout"  # "L2 Regu" # "Dropout"
    partial_l2 = False
    dx_compute_method = "centre"
    output_flow = False
    th_score = 0.8
    for loss_weight_lambda in loss_weight_lambdas:
        test_arguments = TestAruguments(norm, pos_loss_method, th_train_iou, min_test_epoch, max_test_epoch,
                                        dx_compute_method=dx_compute_method, output_flow=output_flow,
                                        score_threshold=th_score,
                                        loss_weight_lambda=loss_weight_lambda, partial_l2_penalty=partial_l2,
                                        prevent_overfitting_method=prevent_overfitting_method)
        test_arguments.show_arguments()
        write_result_path = test_arguments.get_write_result_path()

        all_csv_result = open(write_result_path, "w")
        first_write(all_csv_result)
        th_nms_iou_ls = [0.01]
        th_iou_p_ls = [1]  # [0.6, 0.8, 1]
        for th_iou_p in th_iou_p_ls:
            for th_nms_iou in th_nms_iou_ls:
                file_before = open("Debug Before NMS.txt", "w")
                file_after = open("Debug After NMS.txt", "w")
                test_arguments.th_nms_iou = th_nms_iou
                test_arguments.th_iou_p = th_iou_p
                _test_k_fold(file_before, file_after, test_arguments, all_csv_result)


if __name__ == '__main__':
    main()
