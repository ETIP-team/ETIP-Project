# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2019-04-13

import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from bilstm_attention_crf import AttentionNestedNERModel
from crf_config import Config
from attention_neww2vmodel import geniaDataset
from utils import data_prepare


def train_one_batch(config: Config, model: AttentionNestedNERModel, one_batch_data: list, one_batch_label: list):
    # consider carefully to split for batch
    # partition by length or actually one seq to save.
    batch_loss = []

    # if max_seqs_nested_level > 1 and seq_length > 2:
    #     wait = True
    for seq_index in range(len(one_batch_label)):
        one_seq_word_ids = one_batch_data[seq_index]
        one_seq_labels = one_batch_label[seq_index]
        if config.cuda:
            one_seq_word_ids = Variable(t.Tensor([one_seq_word_ids]).cuda().long())
            one_seq_labels = Variable(t.Tensor([one_seq_labels]).cuda().long())
        else:
            one_seq_word_ids = Variable(t.Tensor(one_seq_word_ids).long())
            one_seq_labels = Variable(t.Tensor(one_seq_labels).long())
        # one_seq_labels = [nested_level, num_batch]
        neg_log_loss = model.neg_log_likelihood(one_seq_word_ids, one_seq_labels)
        # one_batch_loss = model.calc_loss(predict_result, seqs_labels.reshape(-1))
        # print(neg_log_loss)
        batch_loss.append(neg_log_loss.cpu().data.numpy())
        model.optimizer.zero_grad()
        neg_log_loss.backward()
        model.optimizer.step()
    return np.array(batch_loss).mean()


def train_one_epoch(config: Config,
                    model: AttentionNestedNERModel) -> np.ndarray:  # split nested_level then sequence length.

    epoch_losses = []

    # first split by the nested level!

    data_nested_level = np.array([len(item) for item in config.train_label])

    train_nested_level_list = sorted(list(set(data_nested_level)))

    for nested_level in train_nested_level_list:
        sub_data = np.array(config.train_data)[data_nested_level == nested_level]
        sub_label = np.array(config.train_label)[data_nested_level == nested_level]

        # then split by the sentence length.

        sub_data_length = np.array([len(item) for item in sub_data])  # nested levels

        sub_data_length_list = sorted(list(set(sub_data_length)))

        for seq_len in sub_data_length_list:
            sub_sub_data = sub_data[sub_data_length == seq_len]
            sub_sub_label = sub_label[sub_data_length == seq_len]

            num_sub_sub_samples = len(sub_sub_data)
            for left_boundary in range(0, num_sub_sub_samples, config.num_batch):
                right_boundary = left_boundary + config.num_batch if left_boundary + config.num_batch < num_sub_sub_samples else num_sub_sub_samples

                one_batch_data = sub_sub_data[left_boundary: right_boundary]
                one_batch_label = sub_sub_label[left_boundary: right_boundary]
                batch_loss = train_one_batch(config, model, one_batch_data.tolist(), one_batch_label.tolist())
                epoch_losses.append(batch_loss)
    return np.array(epoch_losses).mean()


def start_training(config: Config, model: AttentionNestedNERModel):
    # setting hyper parameter.
    model.optimizer = optim.Adam(params=model.parameters(), lr=config.learning_rate, weight_decay=config.l2_penalty)
    print("Start Training------------------------------------------------", "\n" * 2)
    for epoch in range(config.max_epoch):
        epoch_loss = train_one_epoch(config, model)
        print("Epoch: ", epoch + 1, " " * 5, "Loss: ", epoch_loss)
        if epoch + 1 >= config.start_save_epoch:
            config.save_model(model, epoch)
    return


def main():
    config = Config()
    word_dict = geniaDataset()
    model = AttentionNestedNERModel(config, word_dict).cuda() if config.cuda else AttentionNestedNERModel(config,
                                                                                                          word_dict)

    config.train_data, config.train_str, config.train_label = data_prepare(config, config.get_train_path(), word_dict)

    start_training(config, model)


if __name__ == '__main__':
    main()
