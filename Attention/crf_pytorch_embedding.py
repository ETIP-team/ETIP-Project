import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gensim.models import KeyedVectors
from torch.autograd import Variable

torch.manual_seed(1)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
model_path = './model/pytorch_embedding.pth'

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

tag_to_ix = {"GT": 0, "BG": 1, START_TAG: 2, STOP_TAG: 3}


def read_train_data():
    training_data = []
    with open('./dataset/ace_train.data', 'r') as f:
        all_info = f.read().strip('\n').split('\n\n')
        for infos in all_info:
            infos = infos.split('\n')
            words = infos[0].split()
            word_len = len(words)
            gt_count = np.zeros(word_len)
            labels = []
            rl = [item.split(' G#')[0].split(',') for item in infos[2].split('|')] if len(infos) == 3 else []
            for right, left in rl:
                gt_count[int(right): int(left)] += 1
            for i in range(word_len):
                labels.append('GT' if gt_count[i] > 0 else 'BG')
            training_data.append((words, labels))
    return training_data


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] if w in to_ix else 0 for w in seq]
    return torch.Tensor(idxs).long()


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    """???= torch.log(torch.sum(torch.exp(vec)))"""
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)  # //2->int  /2->float

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                '''next_tag_var[t] = forward_var[t] + trans_score[t->next_tag] + emit_score[next_tag]'''
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        self.hidden = self.hidden
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []  # 记录最可能路径

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

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
                '''
                    forward_var 第i-1步迭代的结果
                    next_tag_var[t] = forward_var[t] + transitions[next_tag <--t]
                    next_tag_var 第i个状态是next_tag的概率（对数）
                    bptrs_t = [best_tag_id], viterbivars_t 最可能路径（best_tag_id->next_tag）及概率
                '''
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            '''feat: emission scores'''
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)
        # lstm_feats.
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


# Make up some training data
training_data = read_train_data()
# training_data = [(
#     "the wall street journal reported today that apple corporation made money".split(),
#     "B I I I O O O B I O O".split()
# ), (
#     "georgia tech is a university in georgia".split(),
#     "B I O O O O B".split()
# )]

word_to_ix = {'UNK': 0}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)


def train():
    # tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Check predictions before training
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
        precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
        # precheck_sent = training_data[0][0]
        print(model(precheck_sent))

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(40):  # again, normally you would NOT do 300 epochs, it is toy data
        print('______________________________')
        print(f'epoch{epoch}:')
        losses = []
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            losses.append(loss.tolist())
            optimizer.step()
        print(f'Avg_loss:{np.mean(losses)}')

    # Check predictions after training
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
        # precheck_sent = training_data[0][0]
        print(model(precheck_sent))
    # We got it!

    torch.save(model.state_dict(), model_path)


def _test():
    infile = open('./dataset/ace_test.data', 'r')

    alldata = infile.read().strip('\n').split('\n\n')
    infile.close()
    x_test = []
    y_test = []
    gts = []

    for data in alldata:
        infos = data.strip('\n').split('\n')
        words = infos[0].split()
        word_len = len(words)
        labels = []
        gt_rl = []
        gt_count = np.zeros(word_len)
        rl = [item.split(' G#')[0].split(',') for item in infos[2].split('|')] if len(infos) == 3 else []
        for right, left in rl:
            gt_count[int(right): int(left)] += 1
            gt_rl.append([int(right), int(left)])
        for i in range(word_len):
            labels.append(0 if gt_count[i] > 0 else 1)
        x_test.append(words)
        y_test.append(labels)
        gts.append(gt_rl)

    crf = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    crf.load_state_dict(torch.load(model_path))
    crf.eval()

    TP = 0
    FN = 0
    confusion_matrix = np.zeros((2, 2))
    for i in range(len(x_test)):
        y_pred = crf(prepare_sequence(x_test[i], word_to_ix))[1]
        for j in range(len(y_pred)):
            confusion_matrix[y_test[i][j], y_pred[j]] += 1
            # if y_test[i][j] == 0 and y_pred[j] == 1:
            #     wait = True
        for right, left in gts[i]:
            try:
                y_pred[right: left].index(1)
                if True:
                    FN += 1
                else:
                    TP += 1
            except ValueError:
                TP += 1
    print(confusion_matrix)
    for i in range(2):
        print(f'{i}\t{confusion_matrix[i, i] / np.sum(confusion_matrix[:, i])}\t'
              f'{confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])}')
    print(f'TP:{TP}\tFN:{FN}\trecall:{TP / (TP + FN)}')


if __name__ == '__main__':
    # train()
    _test()
