# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2018-10-12

import torch as t
import torch.optim as optim
import numpy as np
from model import RCNN
import config as cfg
from arguments import TrainArguments

learning_rate = cfg.LEARNING_RATE
beta = cfg.L2_BETA


def train_batch(sentence, rois, ridx, gt_cls, gt_tbbox, rcnn):
    predict_cls, predict_tbbox = rcnn(sentence, rois, ridx)
    loss, loss_cls, loss_loc = rcnn.calc_loss(predict_cls, predict_tbbox, gt_cls, gt_tbbox)
    _loss = loss.data.cpu().numpy()
    _loss_cls = loss_cls.data.cpu().numpy()
    _loss_loc = loss_loc.data.cpu().numpy()

    # back propagation
    rcnn.optimizer.zero_grad()
    loss.backward()
    rcnn.optimizer.step()
    return _loss, _loss_cls, _loss_loc


def train_epoch(train_arguments, rcnn):
    train_sentences_num = len(train_arguments.train_set)
    perm = np.random.permutation(train_sentences_num)
    perm = train_arguments.train_set[perm]

    losses = []
    losses_cls = []
    losses_loc = []

    for sentence, rois, ridx, gt_cls, gt_tbbox in train_arguments.batch_train_data_generator(perm):
        loss, loss_cls, loss_loc = train_batch(sentence, rois, ridx, gt_cls, gt_tbbox, rcnn)
        # loss = train_batch(sentence, rois, ridx, gt_cls, rcnn, optimizer)
        losses.append(loss)
        losses_cls.append(loss_cls)
        losses_loc.append(loss_loc)

    avg_sum_loss = np.mean(losses)
    avg_loss_cls = np.mean(losses_cls)
    avg_loss_loc = np.mean(losses_loc)
    print(f'Avg loss = {avg_sum_loss:.4f}; loss_cls = {avg_loss_cls:.4f}, loss_loc = {avg_loss_loc:.4f}')


def start_training(train_arguments, folder_index):
    rcnn = RCNN(train_arguments.pos_loss_method, train_arguments.loss_weight_lambda).cuda()

    npz_path = train_arguments.get_train_data_path(folder_index)
    npz = np.load(npz_path)
    print("\n\n\nload from:  ", npz_path)
    train_arguments.train_sentences = npz['train_sentences']
    train_arguments.train_sentence_info = npz['train_sentence_info']
    train_arguments.train_roi = npz['train_roi']
    train_arguments.train_cls = npz['train_cls']
    train_arguments.train_tbbox = npz["train_norm_tbbox"] if train_arguments.normalize else npz['train_tbbox']
    train_arguments.train_sentences = t.Tensor(train_arguments.train_sentences)
    train_arguments.train_set = np.random.permutation(train_arguments.train_sentences.size(0))  # like shuffle
    if train_arguments.prevent_overfitting_method.lower() == "l2 norm":
        if train_arguments.partial_l2_penalty:
            optimizer = optim.Adam(rcnn.conv1.parameters(), lr=train_arguments.learning_rate,
                                   weight_decay=train_arguments.l2_beta)
        else:
            optimizer = optim.Adam(rcnn.parameters(), lr=train_arguments.learning_rate,
                                   weight_decay=train_arguments.l2_beta)
        rcnn.optimizer = optimizer

    for epoch_time in range(train_arguments.max_iter_epoch):
        print('===========================================')
        print('[Training Epoch {}]'.format(epoch_time + 1))

        train_epoch(train_arguments, rcnn)
        if epoch_time >= train_arguments.start_save_epoch:
            save_directory = train_arguments.get_save_directory(folder_index)
            save_path = save_directory + "model_epoch" + str(epoch_time + 1) + ".pth"
            t.save(rcnn.state_dict(), save_path)
            print("Model save in ", save_path)


def train_k_fold():
    start_save_epoch = 30
    th_train_iou = 0.6
    pos_loss_type = "mse"  # lower case
    prevent_overfitting_method = "L2 Norm"
    with_regressor = True
    loss_weight_lambda = 2.0
    norm = True  # False
    partial_l2_penalty = True  # add penalty in conv1 parameters.
    max_iter_epoch = 40

    train_arguments = TrainArguments(start_save_epoch, pos_loss_type, norm, th_train_iou,
                                     max_iter_epoch, prevent_overfitting_method,
                                     with_regressor=with_regressor, partial_l2_penalty=partial_l2_penalty,
                                     loss_weight_lambda=loss_weight_lambda)

    train_arguments.show_arguments()

    for folder_index in range(train_arguments.fold_k):
        start_training(train_arguments, folder_index)


if __name__ == '__main__':
    train_k_fold()
