# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2018-10-12


import os
import torch as t
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from model import RCNN, RCNN_NO_REGRESSOR
import config as cfg

learning_rate = cfg.LEARNING_RATE
beta = cfg.L2_BETA


# def train_batch(sentence, rois, ridx, gt_cls, gt_tbbox, rcnn, optimizer): # gt train
def train_batch(sentence, rois, ridx, gt_cls, rcnn, optimizer):
    predict_cls = rcnn(sentence, rois, ridx)
    loss = rcnn.calc_loss(predict_cls, gt_cls)
    _loss = loss.data.cpu().numpy()

    # back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return _loss


# todo debug
def train_epoch(train_set, train_sentences, train_sentence_info, train_roi, train_cls, train_tbbox, rcnn, optimizer):
    batch_sentence_num = 4  # 2
    roi_num = 64
    pos_roi_num = int(roi_num * 0.25)
    neg_roi_num = roi_num - pos_roi_num
    sentences_num = len(train_set)
    perm = np.random.permutation(sentences_num)
    perm = train_set[perm]

    losses = []
    # losses_cls = []
    # losses_loc = []

    for i in range(0, sentences_num, batch_sentence_num):
        left_boundary = i
        right_boundary = min(i + batch_sentence_num, sentences_num)
        torch_seg = perm[left_boundary:right_boundary]
        sentence = Variable(train_sentences[torch_seg, :, :]).cuda()
        ridx = []  # sentence's rois belong id in one batch
        glo_ids = []

        for j in range(left_boundary, right_boundary):
            info = train_sentence_info[perm[j]]
            pos_idx = info['pos_idx']
            neg_idx = info['neg_idx']
            ids = []

            if len(pos_idx) > 0:
                ids.append(np.random.choice(pos_idx, size=pos_roi_num))
            if len(neg_idx) > 0:
                ids.append(np.random.choice(neg_idx, size=neg_roi_num))
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
        # try:
        rois = train_roi[glo_ids]
        # except:
        #     wait = True
        gt_cls = Variable(t.Tensor(train_cls[glo_ids])).cuda()
        # gt_tbbox = Variable(t.Tensor(train_tbbox[glo_ids])).cuda()

        # loss, loss_cls, loss_loc = train_batch(sentence, rois, ridx, gt_cls, gt_tbbox, rcnn, optimizer)  # gt train
        loss = train_batch(sentence, rois, ridx, gt_cls, rcnn, optimizer)
        losses.append(loss)

    avg_loss = np.mean(losses)
    # avg_loss_sc = np.mean(losses_cls)   # gt train
    # avg_loss_loc = np.mean(losses_loc)   # gt train
    # print(f'Avg loss = {avg_loss:.4f}; loss_sc = {avg_loss_sc:.4f}, loss_loc = {avg_loss_loc:.4f}')   # gt train
    print(f'Avg loss = {avg_loss:.4f}')


def start_training(th_iou, n_epoch, folder, start_save_epoch):
    # rcnn = RCNN().cuda()
    rcnn = RCNN_NO_REGRESSOR().cuda()

    # print(rcnn)
    # npz_path = 'dataset/train/train_data_npz/train_th_iou_'+str(th_iou)+"_norm_"+str(folder)+".npz"
    npz_path = 'dataset/train/train_relabeled_data_npz/relabeled_train_th_iou_' + str(th_iou) + str(folder) + ".npz"
    # npz_path = 'dataset/train/train_relabeled_data_npz/relabeled_train_gt'+str(folder) + ".npz"   # gt train
    npz = np.load(npz_path)
    print("\n\n\nload from:  ", npz_path)
    train_sentences = npz['train_sentences']
    train_sentence_info = npz['train_sentence_info']
    train_roi = npz['train_roi']
    train_cls = npz['train_cls']

    train_tbbox = npz['train_tbbox']

    train_sentences = t.Tensor(train_sentences)

    Ntrain = train_sentences.size(0)
    train_set = np.random.permutation(Ntrain)  # like shuffle

    optimizer = optim.Adam(rcnn.parameters(), lr=learning_rate, weight_decay=beta)

    for i in range(n_epoch):
        print(f'===========================================')
        print(f'[Training Epoch {i+1}]')
        train_epoch(train_set, train_sentences, train_sentence_info, train_roi, train_cls, train_tbbox, rcnn, optimizer)
        if i >= start_save_epoch:
            path = "model/rcnn_jieba/relabeled_gt_model_" + str(th_iou) + "/" + str(folder)
            if not os.path.exists(path):
                os.makedirs(path)
            save_path = path + "/model_epoch" + str(i + 1) + ".pth"
            t.save(rcnn.state_dict(), save_path)
            print("Model save in ", save_path)


def train_k_fold():
    # th_iou_ls = [0.4, 0.5, 0.6]
    th_train_iou = 0.9
    start_save_epoch = 30
    print("For this K fold Training:\n")
    print("It's a GT train!\n")
    print("th_train_iou      ", th_train_iou, "\n\n")

    for i in range(1, cfg.K_FOLD + 1):
        start_training(th_train_iou, 40, i, start_save_epoch)


if __name__ == '__main__':
    train_k_fold()
