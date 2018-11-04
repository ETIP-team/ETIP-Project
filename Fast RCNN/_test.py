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
from model import RCNN, RCNN_NO_REGRESSOR
from utils import reg_to_bbox, non_maximum_suppression, evaluate, get_count_dataframe_by_confusion_matrix, denorm
from utils import first_write, write_one_result

word_embedding_dim = cfg.WORD_EMBEDDING_DIM
classes_num = cfg.CLASSES_NUM
# th_iou_test = cfg.TH_IOU_TEST
# th_iou_nms = cfg.TH_IOU_NMS
th_score = cfg.TH_SCORE


def load_model(model_path, pos_loss_method):
    rcnn = RCNN(pos_loss_method).cuda()
    # rcnn = RCNN_NO_REGRESSOR().cuda()
    rcnn.load_state_dict(t.load(model_path))
    return rcnn


def load_test_sentence(test_sentence_npz_path):
    npz = np.load(test_sentence_npz_path)
    test_sentences = npz['test_sentences']
    test_sentence_info = npz['test_sentence_info']
    test_roi = npz['test_roi']

    test_sentences = t.Tensor(test_sentences).cuda()

    return test_sentences, test_sentence_info, test_roi


def _test_one_sentence(sentence, rois, rcnn, k_fold_index, th_iou_nms, th_train_iou, norm, score_threshold=th_score):
    roi_num = rois.shape[0]
    ridx = np.zeros(roi_num).astype(int)
    pred_cls_score, pred_tbbox = rcnn(sentence, rois, ridx)
    pred_cls_score = F.softmax(pred_cls_score, dim=1).data.cpu().numpy()  # softmax score for each row
    # pred_cls_score = pred_cls_score.data.cpu().numpy()
    pred_tbbox = pred_tbbox.data.cpu().numpy()
    # denorm   18-10-19
    if norm:
        pred_tbbox = denorm(pred_tbbox, k_fold_index, th_train_iou)

    pred_bbox = reg_to_bbox(sentence.size(2), pred_tbbox, rois)

    result_bbox = []
    result_cls = []
    for c in range(1, classes_num + 1):
        c_cls_score = pred_cls_score[:, c]
        c_bboxs = pred_bbox[:, :, c].T

        boxes = non_maximum_suppression(c_cls_score, c_bboxs, iou_threshold=th_iou_nms, score_threshold=score_threshold)
        result_bbox.extend(boxes)
        result_cls.extend([c] * len(boxes))  # print the predict result of this sentence.

    if len(result_cls) == 0:  # if the sentence without detecting anything!, we will lower the score criterion
        for c in range(1, classes_num + 1):
            c_sc = pred_cls_score[:, c]
            c_bboxs = pred_bbox[:, :, c].T
            boxes = non_maximum_suppression(c_sc, c_bboxs, iou_threshold=th_iou_nms,
                                            score_threshold=score_threshold / 2)
            result_bbox.extend(boxes)
            result_cls.extend([c] * len(boxes))
        result_bbox = result_bbox[:1]
        result_cls = result_cls[:1]

    # print(result_cls)
    return np.array(result_bbox), np.array(result_cls)


def gt_test_one_sentence(sentence, rois, rcnn, k_fold_index, th_iou_nms, score_threshold=th_score):
    roi_num = rois.shape[0]
    ridx = np.zeros(roi_num).astype(int)
    pred_cls_score = rcnn(sentence, rois, ridx)
    pred_cls_score = F.softmax(pred_cls_score, dim=1).data.cpu().numpy()  # softmax score for each row
    filt = np.max(pred_cls_score > th_score, axis=1)
    result_cls = np.argmax(pred_cls_score[filt], axis=1)
    rois = rois[filt]
    return result_cls, rois


def _test_epoch(model_path, test_sentence_npz_path, confusion_matrix, k_fold_index, th_iou_nms, th_iou_p,
                pos_loss_method, th_train_iou, norm):  # todo
    rcnn = load_model(model_path, pos_loss_method)
    test_sentences, test_sentence_info, test_roi = load_test_sentence(test_sentence_npz_path)

    sentence_num = test_sentences.size(0)

    perm = np.random.permutation(sentence_num)

    for i in range(sentence_num):
        pi = perm[i]
        sentence = Variable(test_sentences[pi:pi + 1]).cuda()

        info = test_sentence_info[pi]
        idxs = info['roi_ids']
        rois = test_roi[idxs]

        result_bbox, result_cls = _test_one_sentence(sentence, rois, rcnn, k_fold_index, th_iou_nms, th_train_iou, norm)
        # result_cls, rois = gt_test_one_sentence(sentence, rois, rcnn, k_fold_index, th_iou_nms)

        evaluate(result_bbox, result_cls, info, confusion_matrix, th_iou_p)
        # gt_evaluate(rois, result_cls, info, confusion_matrix, th_iou_p)

    return


def _test_k_fold(model_k_folder_path, test_sentence_npz_k_folder_path, th_iou_nms, th_iou_p, all_csv_result,
                 pos_loss_method, th_train_iou, norm):
    for model_index in range(40, 41):
        confusion_matrix = np.zeros((classes_num + 1, classes_num + 1))
        for i in range(1, cfg.K_FOLD + 1):
            model_path = model_k_folder_path + os.listdir(model_k_folder_path)[i-1] + "/model_epoch" + str(
                model_index) + ".pth"
            test_sentence_npz_path = test_sentence_npz_k_folder_path + "test" + str(i) + ".npz"
            _test_epoch(model_path, test_sentence_npz_path, confusion_matrix, i - 1, th_iou_nms, th_iou_p,
                        pos_loss_method, th_train_iou, norm)

        confusion_matrix = confusion_matrix.astype(np.int32).tolist()
        temp_pandas = get_count_dataframe_by_confusion_matrix(confusion_matrix)


        # # todo def !
        # print("Epoch:  ", model_index, "th_iou_nms", th_iou_nms, "th_iou_p", th_iou_p)
        write_one_result(all_csv_result, th_train_iou, th_iou_nms, th_iou_p, temp_pandas)
        print(temp_pandas)
        # all_csv_result.write(
        #     "epoch, " + str(model_index) + ", th_nms, " + str(th_iou_nms) + ", th_p, " + str(th_iou_p) + "\n")
        # print(temp_pandas)
        # temp_pandas.to_csv("./result/insurance/temp.csv")
        # temp_file = open("./result/insurance/temp.csv", "r").read()
        # all_csv_result.write(temp_file)

        # print("\n\n\n\nConfusion Matrix")
        # for i in range(classes_num+1):
        #     print(confusion_matrix[i])

        # print("\n")
        # print("Not hit:", [len(utils.not_hit[i])for i in range(len(utils.not_hit))])
        #
        # print("Hit!!", sum(utils.hit))
        # for i in range(len(utils.not_hit)):
        #     for j in range(len(utils.not_hit[i])):
        #         print(utils.not_hit[i][j])

        # print('Test complete')
    return


def main():
    pos_loss_method = "smoothl1"  # "mse"
    th_train_iou = 0.9  # 0.8
    norm = False  # Flase
    print("For this n condition K fold Testing:\n")
    print("Normalize      ", norm, "\n")
    print("pos_loss_method     ", pos_loss_method, "\n")
    print("th_train_iou      ", th_train_iou, "\n\n\n")
    if norm:
        write_file_path = "./result/insurance/normed_data/norm_"
        model_k_folder_path = "model/rcnn_jieba/relabeled_norm_"
    else:
        write_file_path = "./result/insurance/original_data/"
        model_k_folder_path = "model/rcnn_jieba/relabeled_"

    write_file_path += "train" + str(th_train_iou) + "_" + pos_loss_method + "_train.csv"

    model_k_folder_path += pos_loss_method + "_train_iou_" + str(th_train_iou) + "/"

    print("load model form sum fold", model_k_folder_path)

    all_csv_result = open(write_file_path, "w")
    first_write(all_csv_result)
    th_nms_iou_ls = [0.2]
    th_iou_p_ls = [0.6, 0.8, 1]  # [1]
    for th_iou_p in th_iou_p_ls:
        for th_nms_iou in th_nms_iou_ls:
            test_sentence_npz_k_folder_path = "dataset/test/test_relabeled_data_npz/"
            _test_k_fold(model_k_folder_path, test_sentence_npz_k_folder_path, th_nms_iou, th_iou_p, all_csv_result,
                         pos_loss_method, th_train_iou, norm)


if __name__ == '__main__':
    main()
