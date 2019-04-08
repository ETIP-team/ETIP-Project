# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2019-04-02

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_hidden, bidirectional, num_classes):
        super(BiLSTM_Attention, self).__init__()

        # config:
        self.n_hidden = n_hidden

        # model info:
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=bidirectional)
        self.out = nn.Linear(n_hidden * 2, num_classes)

    # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden * 2,
                                  1)  # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy()  # context : [batch_size, n_hidden * num_directions(=2)]

    def forward(self, X):
        input_x = self.embedding(X)  # input : [batch_size, len_seq, embedding_dim]
        input_x = input_x.permute(1, 0, 2)  # input : [len_seq, batch_size, embedding_dim]

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (h_n, c_n) = self.lstm(input_x)
        output = output.permute(1, 0, 2)  # output : [batch_size, len_seq, n_hidden]
        attn_output, attention = self.attention_net(output, h_n)
        return self.out(attn_output), attention  # model : [batch_size, num_classes], attention : [batch_size, n_step]
