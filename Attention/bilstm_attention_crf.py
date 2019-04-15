# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2019-04-03
# Comment: only support one sentence once a time!

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from crf_config import Config
from attention_neww2vmodel import geniaDataset


def argmax(vec):
    # return the argmax as a python int
    _, idx = t.max(vec, 1)
    return idx.item()


def prepare_sequence(config: Config, seqs, to_ix):
    id_list = []
    for seq in seqs:
        id_list.append([to_ix[w] for w in seq])

    # return torch.tensor(idxs, dtype=torch.long)
    return t.Tensor(id_list).cuda().long() if config.cuda else t.Tensor(id_list).long()


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]  # [num_batch]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.shape[1])  # [num_batch, bio_labels]
    return max_score + t.log(t.sum(t.exp(vec - max_score_broadcast)))


class AttentionNestedNERModel(nn.Module):
    def __init__(self, config: Config, word_dic: geniaDataset):
        super(AttentionNestedNERModel, self).__init__()
        self.config = config
        self.tagset_size = len(self.config.bio_labels)

        # basic setting
        embedding_dim = config.embedding_dim
        hidden_units = config.hidden_units
        encode_bi_flag = config.encode_bi_flag
        encode_num_layers = config.encode_num_layers
        decode_num_layers = config.decode_num_layers
        self.classes_num = config.classes_num
        self.linear_hidden_units = config.linear_hidden_units

        # model setting:
        self.encode_lstm = nn.LSTM(embedding_dim, hidden_units, encode_num_layers, bidirectional=encode_bi_flag)
        self.embedding = nn.Embedding.from_pretrained(word_dic.weight)
        self.embedding.weight.requires_grad = True

        decode_hidden_units = hidden_units * 2 if encode_bi_flag else hidden_units
        # max nested level one hot!
        # * 3 is VNER.
        decode_output = hidden_units * 2
        self.decode_lstm = nn.LSTM(decode_hidden_units * 2 + self.config.max_nested_level, decode_output,
                                   decode_num_layers)

        if self.config.attention_method == "concate":
            self.weight_linear = nn.Linear(decode_hidden_units + embedding_dim, 1)
        elif self.config.attention_method == "general":  # general
            self.weight_linear = nn.Linear(hidden_units * 2, hidden_units * 2, bias=False)
        else:  # dot method. should not go here.
            pass

        # CRF start.  note that bio_labels contain START END tag.
        self.hidden2tag = nn.Linear(decode_output, self.tagset_size)
        self.transitions = nn.Parameter(
            t.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[self.config.bio_labels.index(self.config.START_TAG), :] = -10000
        self.transitions.data[:, self.config.bio_labels.index(self.config.STOP_TAG)] = -10000

        self.hidden = None

        self.optimizer = None
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def _compute_context_input(self, s_i: Variable, h: Variable, time: int) -> Variable:
        """
        To compute context input in time t+1 in different compute type.

        :param s_i: decoder output states in time t. shape: [1, num_batch, hidden_size]
        :param h: encoder shape: [seq_len, num_batch, embedding_dim]
        :param time: time t + 1 index.
        :return:
        """
        # context_input =
        seq_len = h.shape[0]
        num_batch = h.shape[1]
        if self.config.attention_method == "general":
            weight_input = self.weight_linear(h.permute(1, 0, 2))  # [num_batch, seq_len, hidden_size]
            weight_input = t.bmm(weight_input, s_i.permute(1, 2, 0))  # [num_batch, seq_len, 1]
            norm_weight = F.softmax(weight_input, dim=1)  # normalize in dimension seq_len

        elif self.config.attention_method == "dot":

            weight_input = t.bmm(s_i.permute(1, 0, 2), h.permute(1, 2,
                                                                 0))  # [num_batch, 1, hidden_size] [num_batch, emb_dim,seq_len]  = [num_batch, 1, seq_len]
            weight_input = weight_input.permute(0, 2, 1)  # [num_batch, seq_len, 1]
            norm_weight = F.softmax(weight_input, dim=1)

        elif self.config.attention_method == "concate":
            s_compute = s_i.expand(seq_len, num_batch, -1)  # [seq_len, num_batch, hidden_size]
            weight_input = t.cat([s_compute, h], 2)  # [seq_len, num_batch, hidden_size + embedding_dim]
            norm_weight = F.softmax(self.weight_linear(weight_input), dim=0).permute(1, 0,
                                                                                     2)  # [num_batch, seq_len, 1]
        else:
            raise KeyError("attention compute method not right.")

        context_input = t.bmm(h.permute(1, 2, 0),
                              norm_weight)  # [num_batch, embedding_dim, seq_len] [num_batch, seq_len, 1] = [num_batch, embedding_dim, 1]

        h_i = h[time, :, :].unsqueeze(0).permute(1, 2, 0)  # [num_batch, embedding_dim, 1]

        context_input = t.cat([context_input, h_i], 1)  # [num_batch, embedding_dim * 2, seq_len = 1]

        context_input = context_input.permute(2, 0, 1)  # [seq_len = 1, num_batch, embedding * 2]
        return context_input

    def _get_lstm_features(self, seqs):
        """seqs: Tensor for word idx."""

        seqs = self.embedding(seqs).permute(1, 0, 2)  # [seq_len, num_batch, embedding_dim]
        seq_len = seqs.shape[0]
        num_batch = seqs.shape[1]
        decode_num_layers = self.decode_lstm.num_layers
        decode_hidden_size = self.decode_lstm.hidden_size

        h, _ = self.encode_lstm.forward(seqs)  # [seq_len, num_batch, embedding_dim]
        # h = h.permute(1, 0, 2)  # [num_batch, seq_len, embedding_dim]
        # seqs be one cause decode input only one
        # initial i = 0.
        if self.config.cuda:
            s_i = Variable(t.Tensor(np.zeros((decode_num_layers, num_batch, decode_hidden_size))).cuda())
            cell_state_i = Variable(t.Tensor(np.zeros((decode_num_layers, num_batch, decode_hidden_size))).cuda())
            # control_nested_tensor = Variable(
            #     t.Tensor(np.ones((1, num_batch, 1)) * control_nested_level).cuda())
        else:

            s_i = Variable(t.Tensor(np.zeros((decode_num_layers, num_batch, decode_hidden_size))))
            cell_state_i = Variable(t.Tensor(np.zeros((decode_num_layers, num_batch, decode_hidden_size))))
            # control_nested_tensor = Variable(
            #     t.Tensor(np.ones((1, num_batch, 1)) * control_nested_level))
        # hidden_size equals embedding_dim!
        output_list = []
        # decode input seq_len must 1
        for control_nested_level in range(self.config.max_nested_level):
            one_hot_control_nested_np = np.zeros((1, num_batch, self.config.max_nested_level))
            one_hot_control_nested_np[:, :, control_nested_level] = 1
            control_nested_tensor = t.Tensor(
                one_hot_control_nested_np)  # [seq_len = 1, num_batch, one_hot nested_level]
            # control_nested_tensor = t.Tensor(
            #     np.ones((1, num_batch, 1)) * control_nested_level)

            control_nested_tensor = Variable(control_nested_tensor.cuda()) if self.config.cuda else Variable(
                control_nested_tensor)
            one_nested_level_output_list = []
            for context_index in range(seq_len):
                s_compute = s_i[-1].unsqueeze(0)  # get last layer [1, num_batch, hidden_size]

                # todo context_input add control info.

                context_input = self._compute_context_input(s_compute, h,
                                                            context_index)  # [seq_len = 1, num_batch, embedding * 2]

                # include h_decode_s_previous.
                # context_input = t.cat([context_input, s_compute, control_nested_tensor],
                #                       2)  # add in third dim. 4-9 VNER

                context_input = t.cat([context_input, control_nested_tensor], 2)  # add in third dim.

                # save one time output and update the cell state.

                one_time_output, (s_i, cell_state_i) = self.decode_lstm.forward(context_input, (s_i, cell_state_i))
                one_nested_level_output_list.append(one_time_output)
            output_list.append(
                t.cat(one_nested_level_output_list).unsqueeze(0))  # seq_len, batch_num, decode_hidden_size

        output = t.cat(output_list, 0)  # [nested_level, seq_len, batch_num, decode_hidden_size]
        output = self.hidden2tag(output)  # [nested_level, seq_len, batch_num, bio_labels_num]
        return output

    def _score_sentence(self, feats, tags):
        # feats shape [seq_len, targets]

        # Gives the score of a provided tag sequence
        score = t.zeros(1).cuda() if self.config.cuda else t.zeros(1)  # [1]
        start_tensor_tag = t.Tensor([self.config.tag_to_ix[self.config.START_TAG]]).long()
        start_tensor_tag = start_tensor_tag.cuda() if self.config.cuda else start_tensor_tag
        tags = t.cat([start_tensor_tag, tags])
        for i, feat in enumerate(feats):  # i+1, i means i transfer to i+1
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.config.tag_to_ix[self.config.STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):  # todo
        backpointers = []
        # feats shape [seq_len, bio_labels]
        # Initialize the viterbi variables in log space
        init_vvars = t.full((1, self.tagset_size), -10000.).cuda() if self.config.cuda else t.full(
            (1, self.tagset_size), -10000.)
        init_vvars[0][self.config.tag_to_ix[self.config.START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (t.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.config.tag_to_ix[self.config.STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.config.tag_to_ix[self.config.START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def _forward_alg(self, feats):
        # feats shape [seq_len, bio_labels]
        # num_batch = feats.shape[1]
        # Do the forward algorithm to compute the partition function
        init_alphas = t.full((1, self.tagset_size), -10000.).cuda()
        # START_TAG has all of the score.
        init_alphas[0][self.config.tag_to_ix[self.config.START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic back prop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:  # [num_batch, bio_labels]
            alphas_t = []  # The forward tensors at this time step
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # emit_score = feat[next_tag].unsqueeze(1).expand(num_batch, self.tagset_size)  # [1, num_batch]
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                # trans_score = self.transitions[next_tag].unsqueeze(0).expand(num_batch, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = t.cat(alphas_t).view(1, self.tagset_size)  # [num_batch, targets]
        terminal_var = forward_var + self.transitions[
            self.config.tag_to_ix[self.config.STOP_TAG]]  # stop prob. [num_batch, targets]
        alpha = log_sum_exp(terminal_var)  # [num_batch]
        return alpha

    def neg_log_likelihood(self, seq, tags):
        # tags:[nested_level, seq_len]
        feats = self._get_lstm_features(seq).squeeze(2)  # [nested_level, seq_len, bio_labels]
        level_losses = []  # [nested_level, num_batch]
        for nested_level in range(self.config.max_nested_level):
            forward_score = self._forward_alg(feats[nested_level])
            gold_score = self._score_sentence(feats[nested_level], tags[nested_level].squeeze())
            level_losses.append((forward_score - gold_score))

        return t.cat(level_losses).mean()  # single value

    def predict(self, sentence):  # only support one sentence!
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        scores = []
        tag_seqs = []
        for nested_level in range(self.config.max_nested_level):
            score, tag_seq = self._viterbi_decode(lstm_feats[nested_level])
            scores.append(score)
            tag_seqs.append(tag_seq)
        return scores, tag_seqs

# if __name__ == '__main__':
#     model = AttentionNestedNERModel(Config(), None)
#     # path = "/"
