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


class BaseArguments:
    def __init__(self, normalize, fold_k, dx_compute_method, pos_loss_method, prevent_overfitting_method,
                 th_train_iou, loss_weight_lambda=1.0, partial_l2_penalty=True):
        self.normalize = normalize  # type boolean
        self.fold_k = fold_k
        self.dx_compute_method = dx_compute_method
        self.pos_loss_method = pos_loss_method
        self.prevent_overfitting_method = prevent_overfitting_method
        self.partial_l2_penalty = partial_l2_penalty
        self.th_train_iou = th_train_iou
        self.loss_weight_lambda = loss_weight_lambda

    def get_model_path(self, model_epoch):  # todo
        # folder index start from 0
        path = f"model/genia/max_cls_len{cfg.MAX_CLS_LEN}_kernal_size_" + str(
            cfg.KERNEL_LENGTH) + "_pooling_out_" + str(
            cfg.POOLING_OUT)  #
        path += "norm_" + self.dx_compute_method + "_" if self.normalize else ""
        path += self.pos_loss_method + "_"
        if self.prevent_overfitting_method.lower() == "l2 regu":
            path += "partial_l2_" if self.partial_l2_penalty else "all_l2_"
        else:
            path += "dropout_"
        path += "train_iou" + str(
            self.th_train_iou) + "_lambda" + str(self.loss_weight_lambda) + "/"
        path += "model_epoch" + str(model_epoch) + ".pth"
        # if self.mode == "debug":
        #     path = 'model/genia/debug/model_epoch40.pth'
        print("Load Model from")
        print(path)

        return path


class TrainArguments(BaseArguments):
    def __init__(self, start_save_epoch, pos_loss_method, normalize, th_train_iou, max_iter_epoch,
                 prevent_overfitting_method, dx_compute_method, b1_add_b2_div_2=False,
                 fold_k=cfg.K_FOLD, with_regressor=True, partial_l2_penalty=True, dropout_rate=0.5,
                 loss_weight_lambda=1.0, learning_rate=1e-4, l2_beta=1e-3, cuda=True,
                 batch_sentence_num=4, roi_num=64, positive_rate=0.25,
                 ):
        super(TrainArguments, self).__init__(normalize, fold_k, dx_compute_method, pos_loss_method,
                                             prevent_overfitting_method, th_train_iou, loss_weight_lambda,
                                             partial_l2_penalty)
        self.is_train = True
        self.with_regressor = with_regressor
        self.b1_add_b2_div_2 = b1_add_b2_div_2
        self.max_iter_epoch = max_iter_epoch
        self.start_save_epoch = start_save_epoch  # type int
        self.learning_rate = learning_rate
        self.l2_beta = l2_beta
        self.dropout_rate = dropout_rate
        self.cuda = cuda
        self.batch_sentence_num = batch_sentence_num
        self.roi_num = roi_num
        self.pos_roi_num = int(positive_rate * self.roi_num)
        self.neg_roi_num = self.roi_num - self.pos_roi_num

        self.train_set = None
        self.train_sentences = None
        self.train_sentence_info = None
        self.train_roi = None
        self.train_cls = None
        self.train_target_bbox = None

    def show_arguments(self):
        print("\nFor this K fold training:\n") if self.fold_k > 1 else print("For this training:\n")
        print("With Regressor:      ", self.with_regressor, "\n")
        print("Normalize:      ", self.normalize, "\n")
        print("Position Loss Type:      ", self.pos_loss_method.upper(), "\n")
        print("Position Loss Weight Lambda:      ", self.loss_weight_lambda, "\n")
        print("Learning Rate:      ", self.learning_rate, "\n")
        if self.prevent_overfitting_method.lower() == "l2 regu":
            print("Prevent Over fitting Method:      ", "L2 Regulization", "\n")
            print("Partial l2 Penalty:      ", self.partial_l2_penalty, "\n")
            print("L2 Beta:      ", self.l2_beta, "\n")
        if self.prevent_overfitting_method.lower() == "dropout":
            print("Prevent Over fitting Method:      ", "Dropout", "\n")
            print("Dropout Rate:      ", self.dropout_rate, "\n")

        print("Th Train Iou:      ", self.th_train_iou, "\n")
        print("Regression Dx Compute Method:      ", end="")
        print("Left Boundary ", "\n\n") if self.dx_compute_method == "left_boundary" else print("Centre ", "\n\n")

    def get_train_data_path(self):

        return f"dataset/train/train_base_max_cls_len{cfg.MAX_CLS_LEN}_train_th_iou_" + str(self.th_train_iou) + ".npz"

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
            gt_target_bbox = Variable(t.Tensor(self.train_target_bbox[glo_ids])).cuda()
            yield sentence, rois, ridx, gt_cls, gt_target_bbox


class TestArguments(BaseArguments):
    def __init__(self, normalize, pos_loss_method, th_train_iou, min_test_epoch, max_test_epoch,
                 dx_compute_method,
                 loss_weight_lambda=1.0, cuda=True, score_threshold=0.6, dropout_rate=0.5,
                 th_nms_iou=0, th_iou_p=0, partial_l2_penalty=True, prevent_overfitting_method="L2 Regu"
                 , fold_k=cfg.K_FOLD, with_regressor=True, mode="test"):
        super(TestArguments, self).__init__(normalize, fold_k, dx_compute_method, pos_loss_method,
                                            prevent_overfitting_method, th_train_iou, loss_weight_lambda,
                                            partial_l2_penalty)
        self.is_train = False
        self.cuda = cuda
        self.th_nms_iou = th_nms_iou
        self.th_iou_p = th_iou_p
        self.score_threshold = score_threshold
        self.min_test_epoch = min_test_epoch
        self.max_test_epoch = max_test_epoch
        self.with_regressor = with_regressor
        self.dropout_rate = dropout_rate
        self.nms_return_all = True
        self.mode = mode

        self.confusion_matrix = None
        self.overlap_matrix = None
        self.gt_overlap_matrix = None
        self.contain_matrix = None
        self.gt_contain_maxtrix = None
        self.model_path = None
        self.test_sentence_npz_path = None
        self.data_frame = None

        self.result_output = open('./result.data', 'w')

    def get_test_npz_path(self):
        path = f'dataset/test/test_base_max_cls_len{cfg.MAX_CLS_LEN}.npz'
        if self.mode == "debug":
            path = 'dataset/debug/debug_test.npz'

        return path

    def get_write_result_path(self):
        path = f"./result/genia/max_cls_len{cfg.MAX_CLS_LEN}_"
        # path += "2cell_0_"  # 12-24
        # path += "more_project"  # 12-25
        # path += "c_ground_truth"
        # path += "normed_data/global_nega_gap0.2_more_3_"

        path += self.dx_compute_method + "_" if self.normalize else "original_data/"
        # path += "debug"

        path += self.pos_loss_method + "_"
        if self.prevent_overfitting_method.lower() == "l2 regu":
            path += "partial_l2_" if self.partial_l2_penalty else "all_l2_"
        else:
            path += "dropout_"
        path += "train_iou" + str(self.th_train_iou) + "_lambda" + str(self.loss_weight_lambda) + ".csv"
        return path

    def initialize_confusion_matrix(self):
        self.confusion_matrix = np.zeros((cfg.CLASSES_NUM + 1, cfg.CLASSES_NUM + 1))
        self.contain_matrix = np.zeros((cfg.CLASSES_NUM, cfg.CLASSES_NUM))
        self.overlap_matrix = np.zeros((cfg.CLASSES_NUM, cfg.CLASSES_NUM))
        self.gt_contain_matrix = np.zeros((cfg.CLASSES_NUM, cfg.CLASSES_NUM))
        self.gt_overlap_matrix = np.zeros((cfg.CLASSES_NUM, cfg.CLASSES_NUM))
        return

    def write_one_result(self, all_csv_result):
        data_lists = self.data_frame.values.tolist()
        print("\nTh_train_iou:  ", self.th_train_iou, "th_nms_iou", self.th_nms_iou, "th_iou_p", self.th_iou_p)
        all_csv_result.write(
            "th_train_iou= " + str(self.th_train_iou) + ", th_nms= " + str(self.th_nms_iou) + ", th_p= " + str(
                self.th_iou_p) + ",,,,,")
        for d_l in data_lists:
            for d in d_l:
                all_csv_result.write(str(d))
                all_csv_result.write(",")
        all_csv_result.write("\n")

    def show_arguments(self):
        print("Mode : Debug") if self.mode == "debug" else 0
        print("For this n condition K fold Testing:\n") if self.fold_k > 1 else print("For thisTesting:\n")
        print("With Regressor:      ", self.with_regressor, "\n")
        print("Normalize:      ", self.normalize, "\n")
        print("Position Loss Type:      ", self.pos_loss_method.upper(), "\n")
        print("Position Loss Weight Lambda:      ", self.loss_weight_lambda, "\n")
        if self.prevent_overfitting_method.lower() == "l2 regu":
            print("Prevent Over Fitting Method:      ", "L2 Regulization", "\n")
            print("Partial l2 Penalty:      ", self.partial_l2_penalty, "\n")
        if self.prevent_overfitting_method.lower() == "dropout":
            print("Prevent Over Fitting Method:      ", "Dropout", "\n")
            print("Dropout Rate:      ", self.dropout_rate, "\n")
        print("Th Train Iou:      ", self.th_train_iou, "\n")
        print("Regression Dx Compute Method:      ", end="")
        print("Left Boundary ", "\n\n") if self.dx_compute_method == "left_boundary" else print("Centre ", "\n\n")
