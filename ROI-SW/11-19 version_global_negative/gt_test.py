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
import utils
from utils import reg_to_bbox, evaluate, get_count_dataframe_by_confusion_matrix, denorm, \
    gt_evaluate, non_maximum_suppression
from utils import first_write, write_one_result

word_embedding_dim = cfg.WORD_EMBEDDING_DIM
classes_num = cfg.CLASSES_NUM
# th_iou_test = cfg.TH_IOU_TEST
# th_iou_nms = cfg.TH_IOU_NMS
th_score = cfg.TH_SCORE


def load_model(model_path):
    rcnn = RCNN_NO_REGRESSOR().cuda()
    rcnn.load_state_dict(t.load(model_path))
    return rcnn


def load_test_sentence(test_sentence_npz_path):
    npz = np.load(test_sentence_npz_path)
    test_sentences = npz['test_sentences']
    test_sentence_info = npz['test_sentence_info']
    test_roi = npz['test_roi']

    test_sentences = t.Tensor(test_sentences).cuda()

    return test_sentences, test_sentence_info, test_roi


def gt_test_one_sentence(sentence, rois, rcnn, k_fold_index, th_iou_nms, score_threshold=th_score):
    roi_num = rois.shape[0]
    ridx = np.zeros(roi_num).astype(int)
    pred_cls_score = rcnn(sentence, rois, ridx)
    pred_cls_score = F.softmax(pred_cls_score, dim=1).data.cpu().numpy()  # softmax score for each row
    # filt = np.max(pred_cls_score > th_score, axis=1)
    # result_cls = np.argmax(pred_cls_score[filt], axis=1)
    result_cls = np.argmax(pred_cls_score, axis=1)
    # rois = rois[filt]
    # I think still need to do nms

    new_cbbox = []
    for index in range(len(result_cls)):
        position_info = []
        if result_cls[index] != 0:
            for add_ in range(result_cls[index]):
                position_info.append(np.zeros(2).tolist())
            position_info.append(rois[index].tolist())
            for add_ in range(classes_num - result_cls[index]):
                position_info.append(np.zeros(2).tolist())
            new_cbbox.append(position_info)
        else:
            new_cbbox.append(np.zeros(((classes_num + 1), 2)).tolist())
    new_cbbox = np.array(new_cbbox)
    # result_cls.
    result_bbox = []
    result_cls = []
    for c in range(1, classes_num + 1):
        c_cls_score = pred_cls_score[:, c]
        c_bboxs = new_cbbox[:, c, :]

        boxes = non_maximum_suppression(c_cls_score, c_bboxs, iou_threshold=th_iou_nms, score_threshold=score_threshold)
        result_bbox.extend(boxes)
        result_cls.extend([c] * len(boxes))  # print the predict result of this sentence.

    return np.array(result_bbox), np.array(result_cls)


def _test_epoch(model_path, test_sentence_npz_path, confusion_matrix, k_fold_index, th_iou_nms, th_iou_p):  # todo
    rcnn = load_model(model_path)
    test_sentences, test_sentence_info, test_roi = load_test_sentence(test_sentence_npz_path)

    sentence_num = test_sentences.size(0)

    perm = np.random.permutation(sentence_num)

    for i in range(sentence_num):
        pi = perm[i]
        sentence = Variable(test_sentences[pi:pi + 1]).cuda()

        info = test_sentence_info[pi]
        idxs = info['roi_ids']
        rois = test_roi[idxs]

        # result_bbox, result_cls = _test_one_sentence(sentence, rois, rcnn, k_fold_index, th_iou_nms)
        result_bbox, result_cls = gt_test_one_sentence(sentence, rois, rcnn, k_fold_index, th_iou_nms)

        evaluate(result_bbox, result_cls, info, confusion_matrix, th_iou_p)
        # gt_evaluate(rois, result_cls, info, confusion_matrix, th_iou_p)

    return


def _test_k_fold(model_k_folder_path, test_sentence_npz_k_folder_path, th_iou_nms, th_iou_p, all_csv_result):
    for model_index in range(40, 41):
        confusion_matrix = np.zeros((classes_num + 1, classes_num + 1))
        for i in range(1, cfg.K_FOLD + 1):
            model_path = model_k_folder_path + os.listdir(model_k_folder_path)[i-1] + "/model_epoch" + str(
                model_index) + ".pth"
            # print("Load model from", model_path)
            test_sentence_npz_path = test_sentence_npz_k_folder_path + "test" + str(i) + ".npz"
            _test_epoch(model_path, test_sentence_npz_path, confusion_matrix, i - 1, th_iou_nms, th_iou_p)

        confusion_matrix = confusion_matrix.astype(np.int32).tolist()
        temp_pandas = get_count_dataframe_by_confusion_matrix(confusion_matrix)

        write_one_result(all_csv_result, model_index, th_iou_nms, th_iou_p, temp_pandas)
        # print("Epoch:  ", model_index, "th_iou_nms", th_iou_nms, "th_iou_p", th_iou_p)
        print(temp_pandas)
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
    th_train_iou = 1
    write_file_path = "./result/insurance/normed_data/gt_train" + str(th_train_iou) + ".csv"
    all_csv_result = open(write_file_path, "w")
    first_write(all_csv_result)
    # th_iou_ls = [0.4, 0.5]
    th_nms_iou_ls = [0.2]
    # th_nms_iou_ls = [0.2]
    th_iou_p_ls = [0.6, 0.8, 1]  # [1]  #
    # for th_iou in th_iou_ls:
    print("For this K fold test:\n")
    print("th_train_iou:     ", th_train_iou, "\n\n")
    for th_iou_p in th_iou_p_ls:
        for th_nms_iou in th_nms_iou_ls:
            model_k_folder_path = "model/rcnn_jieba/relabeled_gt_model_" + str(th_train_iou) + "/"  # gt train
            test_sentence_npz_k_folder_path = "dataset/test/test_relabeled_data_npz/"
            _test_k_fold(model_k_folder_path, test_sentence_npz_k_folder_path, th_nms_iou, th_iou_p, all_csv_result)


if __name__ == '__main__':
    main()
