# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2018-10-03

import torch as t
import torch.nn as nn
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
        right_boundary = rois[:, 1]

        x1 = left_boundary
        x2 = right_boundary
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
        then two full connected and softmax layer for class score
                two full connected layer for loc detection."""

    def __init__(self, pos_loss_method="smoothl1"):
        super(RCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=feature_maps_number,
                      kernel_size=(kernal_length, word_embedding_dim),
                      stride=1,
                      padding=(1, 0)
                      ),
            nn.ReLU(),
        )
        self.roi_pool = ROIPooling(output_size=(1, pooling_out))
        self.flatten_feature = feature_maps_number * pooling_out
        self.cls_fc1 = nn.Linear(self.flatten_feature, self.flatten_feature)
        self.cls_score = nn.Linear(self.flatten_feature, classes_num + 1)

        self.bbox_fc1 = nn.Linear(self.flatten_feature, self.flatten_feature)
        # attention there only 2* (classes_num+1)!
        self.bbox = nn.Linear(self.flatten_feature, 2 * (classes_num + 1))

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.mse_loss = nn.MSELoss()  # modify the loss calculation
        self.pos_loss_method = pos_loss_method

    def forward(self, sentence, rois, ridx):  # todo a little
        sentence = sentence.float()
        result = self.conv1(sentence)
        result = self.roi_pool(result, rois, ridx)
        result = result.view(result.size(0), -1)

        # output
        # compute class:
        cls_softmax_score = self.cls_fc1(result)
        cls_softmax_score = self.cls_score(cls_softmax_score)

        # auto implement in cross entropy
        # cls_softmax_score = F.softmax(cls_softmax_score)

        # compute bbox
        bbox = self.bbox_fc1(result)
        bbox = self.bbox(bbox).view(-1, classes_num + 1, 2)

        return cls_softmax_score, bbox

    def calc_loss(self, probs, bbox, labels, gt_bbox, lmb=1.0):
        labels = labels.long()
        loss_cls = self.cross_entropy_loss(probs, labels)
        lbl = labels.view(-1, 1, 1).expand(labels.size(0), 1, 2)
        mask = (labels != 0).float().view(-1, 1).expand(labels.size(0), 2)
        # todo
        if self.pos_loss_method == "smoothl1":
            loss_loc = self.smooth_l1_loss(bbox.gather(1, lbl).squeeze(1) * mask, gt_bbox.float() * mask)
            # print(self.pos_loss_method)
        else:
            loss_loc = self.mse_loss(bbox.gather(1, lbl).squeeze(1) * mask, gt_bbox.float() * mask)
        loss = loss_cls + lmb * loss_loc
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

        self.roi_pool = ROIPooling(output_size=(1, pooling_out))
        self.flatten_feature = feature_maps_number * pooling_out
        self.cls_fc1 = nn.Linear(self.flatten_feature, self.flatten_feature)
        self.cls_score = nn.Linear(self.flatten_feature, classes_num + 1)
        #
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
    print(rcnn)

#
# def train_batch(sentence, rois, ridx, gt_cls, gt_tbbox, rcnn, optimizer):
#     predict_cls, predict_tbbox = rcnn(sentence, rois, ridx)
#     loss, loss_cls, loss_loc = rcnn.calc_loss(predict_cls, predict_tbbox, gt_cls, gt_tbbox)
#
#     _loss = loss.data.cpu().numpy()
#     _loss_cls = loss_cls.data.cpu().numpy()
#     _loss_loc = loss_loc.data.cpu().numpy()
#
#     # back propagation
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     # return fl, fl_sc, fl_loc
#     return _loss, _loss_cls, _loss_loc
#
#
# def train_epoch(train_set, train_sentences, train_sentence_info, train_roi, train_cls, train_tbbox, rcnn, optimizer):
#     batch_sentence_num = 2
#     roi_num = 64
#     pos_roi_num = int(roi_num*0.25)
#     neg_roi_num = roi_num - pos_roi_num
#     sentences_num = len(train_set)
#     perm = np.random.permutation(sentences_num)
#     perm = train_set[perm]
#
#     losses = []
#     losses_cls = []
#     losses_loc = []
#
#     for i in range(0, sentences_num, batch_sentence_num):
#         left_boundary = i
#         right_boundary = min(i+batch_sentence_num, sentences_num)
#         torch_seg = perm[left_boundary:right_boundary]
#         sentence = Variable(train_sentences[torch_seg, :, :]).cuda()
#         ridx = []  # sentence's rois belong id in one batch
#         glo_ids = []
#
#         for j in range(left_boundary, right_boundary):
#             info = train_sentence_info[perm[j]]
#             pos_idx = info['pos_idx']
#             neg_idx = info['neg_idx']
#             ids = []
#
#             if len(pos_idx) > 0:
#                 ids.append(np.random.choice(pos_idx, size=pos_roi_num))
#             if len(neg_idx) > 0:
#                 ids.append(np.random.choice(neg_idx, size=neg_roi_num))
#             if len(ids) == 0:
#                 continue
#             ids = np.concatenate(ids, axis=0)
#             glo_ids.append(ids)
#             ridx += [j-left_boundary] * ids.shape[0]
#
#         if len(ridx) == 0:
#             continue
#         glo_ids = np.concatenate(glo_ids, axis=0)  # id in all sentences!
#         ridx = np.array(ridx)
#
#         rois = train_roi[glo_ids]
#         gt_cls = Variable(t.Tensor(train_cls[glo_ids])).cuda()
#         gt_tbbox = Variable(t.Tensor(train_tbbox[glo_ids])).cuda()
#
#         loss, loss_cls, loss_loc = train_batch(sentence, rois, ridx, gt_cls, gt_tbbox, rcnn, optimizer)
#         losses.append(loss)
#         losses_cls.append(loss_cls)
#         losses_loc.append(loss_loc)
#
#     avg_loss = np.mean(losses)
#     avg_loss_sc = np.mean(losses_cls)
#     avg_loss_loc = np.mean(losses_loc)
#     print(f'Avg loss = {avg_loss:.4f}; loss_sc = {avg_loss_sc:.4f}, loss_loc = {avg_loss_loc:.4f}')
#
#
# def start_training(n_epoch=10, folder=1):
#     rcnn = RCNN().cuda()
#     # print(rcnn)
#
#     npz = np.load('dataset/train/train_data_npz/train'+str(folder)+".npz")
#     print("load from:  ", 'dataset/train/train_data_npz/train'+str(folder)+".npz")
#     train_sentences = npz['train_sentences']
#     train_sentence_info = npz['train_sentence_info']
#     train_roi = npz['train_roi']
#     train_cls = npz['train_cls']
#     train_tbbox = npz['train_tbbox']
#
#     train_sentences = t.Tensor(train_sentences)
#     # print("train_sentences", train_sentences.shape)
#     # print("type(train_sentences)", type(train_sentences))
#     # print("train_sentences.type", train_sentences.type())
#
#     Ntrain = train_sentences.size(0)
#     train_set = np.random.permutation(Ntrain)  # like shuffle
#
#     optimizer = optim.Adam(rcnn.parameters(), lr=learning_rate, weight_decay=beta)
#
#     for i in range(n_epoch):
#         print(f'===========================================')
#         print(f'[Training Epoch {i+1}]')
#         train_epoch(train_set, train_sentences, train_sentence_info, train_roi, train_cls, train_tbbox, rcnn, optimizer)
#         if i >= 5:
#             t.save(rcnn.state_dict(), "model/rcnn_jieba/"+str(folder)+"/model_epoch"+str(i)+".pth")
#
#
# def train_k_fold():
#     for i in range(1, cfg.K_FOLD+1):
#         start_training(10, i)
#
#
# train_k_fold()
