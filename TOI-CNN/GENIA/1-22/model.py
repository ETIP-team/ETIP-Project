# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2018-10-03

import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import config as cfg

word_embedding_dim = cfg.WORD_EMBEDDING_DIM
feature_maps_number = cfg.FEATURE_MAPS_NUM
kernal_length = cfg.KERNAL_LENGTH
pooling_out = cfg.POOLING_OUT
classes_num = cfg.CLASSES_NUM


class ROIPooling(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.roi_pooling = nn.AdaptiveMaxPool2d(output_size)
        self.size = output_size

    def forward(self, sentences, rois, roi_idx):
        n = rois.shape[0]
        left_boundary = rois[:, 0]
        right_boundary = rois[:, 1] + 1

        x1 = left_boundary
        x2 = right_boundary

        # modify roi projection.
        # x1 = np.maximum(left_boundary - 1, 0)
        # x2 = np.minimum(right_boundary + 1, sentences.shape[2])
        y1 = np.zeros((n, 1), dtype=int)
        y2 = np.ones((n, 1), dtype=int)

        res = []
        for i in range(n):  # for every roi search it's sentence belong to and start pooling
            sentence = sentences[roi_idx[i]].unsqueeze(0)  # use sentence which i_th roi belong to
            sentence = sentence[:, :, int(x1[i]):int(x2[i]), int(y1[i]):int(y2[i])]
            sentence = self.roi_pooling(sentence)
            res.append(sentence)
        res = t.cat(res, dim=0)
        return res


class RCNN(nn.Module):  # todo result connect one more full connected layer. to be declared!
    """A RCNN for named entity recognition.
        Uses an embedding layer, followed by a convolutional, roi pooling layer
        then two fully connected and softmax layer for class score
                two fully connected layer for loc detection."""

    def __init__(self, pos_loss_method="smoothl1", lambd=1.0, prevent_over_fitting_method="l2_penalty"):
        super(RCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=feature_maps_number,
                      kernel_size=(kernal_length, word_embedding_dim),  # kernal size
                      stride=1,
                      padding=(int(kernal_length / 2), 0)
                      ),
            nn.ReLU(),
        )

        # self.roi_pool = ROIPooling(output_size=(pooling_out, 1))
        self.roi_pool = ROIPooling(output_size=(pooling_out, 1))
        self.flatten_feature = feature_maps_number * pooling_out
        self.cls_fc1 = nn.Linear(self.flatten_feature, self.flatten_feature)
        self.cls_score = nn.Linear(self.flatten_feature, classes_num + 1)
        # self.cls_dropout =
        self.bbox_fc1 = nn.Linear(self.flatten_feature, self.flatten_feature)
        # attention there only 2* (classes_num+1)!
        self.bbox = nn.Linear(self.flatten_feature, 2 * (classes_num + 1))

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.mse_loss = nn.MSELoss()  # modify the loss calculation

        self.pos_loss_method = pos_loss_method
        self.prevent_over_fitting_method = prevent_over_fitting_method
        self.lambd = lambd
        self.optimizer = None

    def forward(self, sentence, rois, ridx, dropout_rate=0.5):
        sentence = sentence.float()
        result = self.conv1(sentence)
        result = self.roi_pool(result, rois, ridx)
        result = result.view(result.size(0), -1)

        # output
        # compute class:
        cls_softmax_score = self.cls_fc1(result)
        if self.prevent_over_fitting_method.lower() == "dropout" and self.training:
            cls_softmax_score = F.dropout(cls_softmax_score, p=dropout_rate, training=self.training)

        cls_softmax_score = self.cls_score(cls_softmax_score)

        # auto implement in cross entropy

        # compute bbox
        bbox = self.bbox_fc1(result)
        if self.prevent_over_fitting_method == "dropout" and self.training:
            bbox = F.dropout(bbox, p=dropout_rate, training=self.training)

        bbox = self.bbox(bbox).view(-1, classes_num + 1, 2)

        return cls_softmax_score, bbox

    def calc_loss(self, probs, bbox, labels, gt_bbox):
        labels = labels.long()
        loss_cls = self.cross_entropy_loss(probs, labels)
        lbl = labels.view(-1, 1, 1).expand(labels.size(0), 1, 2)
        mask = (labels != 0).float().view(-1, 1).expand(labels.size(0), 2)
        # todo
        if self.pos_loss_method == "smoothl1":
            loss_loc = self.smooth_l1_loss(bbox.gather(1, lbl).squeeze(1) * mask, gt_bbox.float() * mask)
            # print(self.pos_loss_method)
        elif self.pos_loss_method == "mse":
            loss_loc = self.mse_loss(bbox.gather(1, lbl).squeeze(1) * mask, gt_bbox.float() * mask)
        loss = loss_cls + self.lambd * loss_loc
        return loss, loss_cls, loss_loc


class RCNN_NO_REGRESSOR(nn.Module):  # todo result connect one more full connected layer. to be declared!
    """A RCNN for named entity recognition.
        Uses an embedding layer, followed by a convolutional, roi pooling layer
        then two full connected and softmax layer for class score
                two full connected layer for loc detection."""

    def __init__(self):
        super(RCNN_NO_REGRESSOR, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=feature_maps_number,
                      kernel_size=(kernal_length, word_embedding_dim),
                      stride=1,
                      padding=(1, 0)
                      ),
            nn.ReLU(),
        )

        self.roi_pool = ROIPooling(output_size=(pooling_out, 1))
        self.flatten_feature = feature_maps_number * pooling_out
        self.cls_fc1 = nn.Linear(self.flatten_feature, self.flatten_feature)
        self.cls_score = nn.Linear(self.flatten_feature, classes_num + 1)
        # self.bbox_fc1 = nn.Linear(self.flatten_feature, self.flatten_feature)
        # # attention there only 2* (classes_num+1)!
        # self.bbox = nn.Linear(self.flatten_feature, 2*(classes_num+1))

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        # self.smooth_l1_loss = nn.SmoothL1Loss()
        # self.mse_loss = nn.MSELoss()   # modify the loss calculation

    def forward(self, sentence, rois, ridx):  # todo a little
        sentence = sentence.float()
        result = self.conv1(sentence)
        result = self.roi_pool(result, rois, ridx)
        result = result.view(result.size(0), -1)

        # output
        # compute class:
        cls_softmax_score = self.cls_fc1(result)
        cls_softmax_score = self.cls_score(cls_softmax_score)

        return cls_softmax_score

    def calc_loss(self, probs, labels):
        labels = labels.long()
        loss_cls = self.cross_entropy_loss(probs, labels)
        # lbl = labels.view(-1, 1, 1).expand(labels.size(0), 1, 2)
        # mask = (labels != 0).float().view(-1, 1).expand(labels.size(0), 2)
        # todo
        # loss_loc = self.smooth_l1_loss(bbox.gather(1, lbl).squeeze(1) * mask, gt_bbox.float() * mask)
        # loss_loc = self.mse_loss(bbox.gather(1, lbl).squeeze(1) * mask, gt_bbox.float() * mask)
        return loss_cls


if __name__ == '__main__':
    rcnn = RCNN()
    # rcnn.optimizer = optim.Adam(rcnn.conv1.parameters(), lr=1e-4, weight_decay=1e-3)
    for name, para in rcnn.named_parameters():
        print("name", name)
        # print("parameter", para)
