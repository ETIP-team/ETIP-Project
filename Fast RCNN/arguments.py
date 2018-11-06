# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2018-11-05

import os
import numpy as np
import torch as t
from torch.autograd import Variable
import config as cfg


class BaseArguments(object):
    def __init__(self, normalize, fold_k):
        self.normalize = normalize  # type boolean
        self.fold_k = fold_k


class TrainArguments(BaseArguments):
    def __init__(self, start_save_epoch, pos_loss_method, normalize, th_train_iou, max_iter_epoch,
                 prevent_overfitting_method,
                 fold_k=5, with_regressor=True, partial_l2_penalty=True, dropout_rate=0.5,
                 loss_weight_lambda=1.0, learning_rate=1e-4, l2_beta=1e-3, cuda=True,
                 batch_sentence_num=4, roi_num=64, positive_rate=0.25
                 ):
        super(TrainArguments, self).__init__(normalize, fold_k)
        self.is_train = True
        self.with_regressor = with_regressor
        # train hyper parameters
        self.prevent_overfitting_method = prevent_overfitting_method
        # self.dropout = True  # if dr
        self.max_iter_epoch = max_iter_epoch
        self.start_save_epoch = start_save_epoch  # type int
        self.pos_loss_method = pos_loss_method  # type str
        self.loss_weight_lambda = loss_weight_lambda  # type float
        self.th_train_iou = th_train_iou  # type float
        self.learning_rate = learning_rate
        self.l2_beta = l2_beta
        self.partial_l2_penalty = partial_l2_penalty
        self.dropout_rate = dropout_rate
        self.cuda = cuda
        self.batch_sentence_num = batch_sentence_num
        self.roi_num = roi_num
        self.pos_roi_num = int(positive_rate * self.roi_num)
        self.neg_roi_num = self.roi_num - self.pos_roi_num

        # train data info
        self.train_set = None
        self.train_sentences = None
        self.train_sentence_info = None
        self.train_roi = None
        self.train_cls = None
        self.train_tbbox = None

    def show_arguments(self):
        print("\nFor this K fold training:\n") if self.fold_k > 1 else print("For this training:\n")
        print("With Regressor:      ", self.with_regressor, "\n")
        print("Normalize:      ", self.normalize, "\n")
        print("Position Loss Type:      ", self.pos_loss_method.upper(), "\n")
        print("Position Loss Weight Lambda:      ", self.loss_weight_lambda, "\n")
        print("Learning Rate:      ", self.learning_rate, "\n")
        if self.prevent_overfitting_method.lower() == "l2 regu":
            print("Prevent Over fitting method:      ", "L2 Regulization", "\n")
            print("Partial l2 Penalty:      ", self.partial_l2_penalty, "\n")
            print("L2 Beta:      ", self.l2_beta, "\n")
        if self.prevent_overfitting_method.lower() == "dropout":
            print("Prevent Over fitting method:      ", "Dropout", "\n")
            print("Dropout Rate:      ", self.dropout_rate, "\n")

        print("Th Train Iou      ", self.th_train_iou, "\n\n")

    def get_save_directory(self, folder_index):
        # folder index start from 0
        path = "model/rcnn_jieba/"
        path += "norm_" if self.normalize else ""
        path += self.pos_loss_method + "_"
        if self.prevent_overfitting_method.lower() == "l2 regu":
            path += "partial_l2_" if self.partial_l2_penalty else "all_l2_"
        else:
            path += "dropout_"
        path += "train_iou" + str(
            self.th_train_iou) + "_lambda" + str(self.loss_weight_lambda) + "/" + str(folder_index + 1) + "/"
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_train_data_path(self, folder_index):
        # folder index start from 0
        return "dataset/train/train_relabeled_data_npz/train_th_iou_" + str(self.th_train_iou) + str(
            folder_index + 1) + ".npz"

    def batch_train_data_generator(self, perm):
        train_sentences_num = len(self.train_set)
        for i in range(0, train_sentences_num, self.batch_sentence_num):
            left_boundary = i
            right_boundary = min(i + self.batch_sentence_num, train_sentences_num)
            torch_seg = perm[left_boundary:right_boundary]
            sentence = Variable(self.train_sentences[torch_seg, :, :]).cuda()
            ridx = []  # sentence's rois belong id in one batch
            glo_ids = []

            for j in range(left_boundary, right_boundary):
                info = self.train_sentence_info[perm[j]]
                pos_idx = info['pos_idx']
                neg_idx = info['neg_idx']
                ids = []

                if len(pos_idx) > 0:
                    ids.append(np.random.choice(pos_idx, size=self.pos_roi_num))
                if len(neg_idx) > 0:
                    ids.append(np.random.choice(neg_idx, size=self.neg_roi_num))
                if len(ids) == 0:
                    continue

                # all roi  18-10-19
                # ids.append(pos_idx)
                # ids.append(neg_idx)
                # all roi

                ids = np.concatenate(ids, axis=0)
                glo_ids.append(ids)
                ridx += [j - left_boundary] * ids.shape[0]

            if len(ridx) == 0:
                continue
            glo_ids = np.concatenate(glo_ids, axis=0).astype(np.int64)  # id in all sentences!
            ridx = np.array(ridx)
            rois = self.train_roi[glo_ids]
            gt_cls = Variable(t.Tensor(self.train_cls[glo_ids])).cuda()
            gt_tbbox = Variable(t.Tensor(self.train_tbbox[glo_ids])).cuda()
            yield sentence, rois, ridx, gt_cls, gt_tbbox


class TestAruguments(BaseArguments):
    def __init__(self, normalize, pos_loss_method, th_train_iou, min_test_epoch, max_test_epoch,
                 loss_weight_lambda=1.0, cuda=True, score_threshold=0.6, dropout_rate=0.5,
                 th_nms_iou=0, th_iou_p=0, partial_l2_penalty=True, prevent_overfitting_method="L2 Norm"
                 , fold_k=5, with_regressor=True):
        super(TestAruguments, self).__init__(normalize, fold_k)
        self.is_train = False
        self.cuda = cuda
        self.pos_loss_method = pos_loss_method
        self.loss_weight_lambda = loss_weight_lambda
        self.th_train_iou = th_train_iou  # to get the model path
        self.th_nms_iou = th_nms_iou
        self.th_iou_p = th_iou_p
        self.partial_l2_penalty = partial_l2_penalty
        self.prevent_overfitting_method = prevent_overfitting_method
        self.score_threshold = score_threshold
        self.min_test_epoch = min_test_epoch
        self.max_test_epoch = max_test_epoch
        self.with_regressor = with_regressor
        self.dropout_rate = dropout_rate

        self.confusion_matrix = None
        self.model_path = None
        self.test_sentence_npz_path = None
        self.data_frame = None

    def get_model_path(self, folder_index, model_epoch):  # todo
        # folder index start from 0
        path = "model/rcnn_jieba/"
        path += "norm_" if self.normalize else ""
        path += self.pos_loss_method + "_"
        if self.prevent_overfitting_method.lower() == "l2 regu":
            path += "partial_l2_" if self.partial_l2_penalty else "all_l2_"
        else:
            path += "dropout_"
        path += "train_iou" + str(self.th_train_iou) + "_lambda" + str(self.loss_weight_lambda) + "/"
        path += str(folder_index + 1) + "/"
        path += "model_epoch" + str(model_epoch) + ".pth"

        return path

    def get_test_npz_path(self, folder_index):  # todo
        # folder index start from 0
        path = "dataset/test/test_relabeled_data_npz/test" + str(folder_index + 1) + ".npz"
        return path

    def get_write_result_path(self):  # todo
        path = "./result/insurance/"
        path += "normed_data/" if self.normalize else "original_data/"
        path += self.pos_loss_method + "_"
        if self.prevent_overfitting_method.lower() == "l2 regu":
            path += "partial_l2_" if self.partial_l2_penalty else "all_l2_"
        else:
            path += "dropout_"
        path += "train_iou" + str(self.th_train_iou) + "_lambda" + str(self.loss_weight_lambda) + ".csv"
        return path

    def initialize_confusion_matrix(self):
        self.confusion_matrix = np.zeros((cfg.CLASSES_NUM + 1, cfg.CLASSES_NUM + 1))
        return

    def write_one_result(self, all_csv_result):
        data_lists = self.data_frame.values.tolist()
        print("\nTh_train_iou:  ", self.th_train_iou, "th_nms_iou", self.th_nms_iou, "th_iou_p", self.th_iou_p)
        all_csv_result.write(
            "th_train_iou= " + str(self.th_train_iou) + ", th_nms= " + str(self.th_nms_iou) + ", th_p= " + str(
                self.th_iou_p) + ",,")
        for d_l in data_lists:
            for d in d_l:
                all_csv_result.write(str(d))
                all_csv_result.write(",")
        all_csv_result.write("\n")

    def show_arguments(self):
        print("For this n condition K fold Testing:\n") if self.fold_k > 1 else print("For thisTesting:\n")
        print("With Regressor:      ", self.with_regressor, "\n")
        print("Normalize:      ", self.normalize, "\n")
        print("Position Loss Type:      ", self.pos_loss_method.upper(), "\n")
        print("Position Loss Weight Lambda:      ", self.loss_weight_lambda, "\n")
        if self.prevent_overfitting_method.lower() == "l2 regu":
            print("Prevent Over fitting method:      ", "L2 Regulization", "\n")
            print("Partial l2 Penalty:      ", self.partial_l2_penalty, "\n")
        if self.prevent_overfitting_method.lower() == "dropout":
            print("Prevent Over fitting method:      ", "Dropout", "\n")
            print("Dropout Rate:      ", self.dropout_rate, "\n")
        print("Th Train Iou      ", self.th_train_iou, "\n\n")
