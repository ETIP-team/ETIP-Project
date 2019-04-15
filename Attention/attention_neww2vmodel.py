import gensim
import re
# from torch.utils.data import DataLoader
import torch
from config import Config


class geniaDataset:
    def __init__(self):
        self.config = Config()

        self.w2vmodel = w2VModel(self.config.WORD_VEC_MODEL_PATH)
        self.embDim = 200
        self.aim = 'train'  # train or test.
        # our vocabulary dictionary, dictionary value is 0 means padding idx.
        self.cateVocab = {}
        # self.path = self.config.get_train_path()
        # self.path1 = self.config.TEST_FILE_PATH
        # our vocabulary dictionary, dictionary value is 0 means padding idx.
        self.vocabToOld = {}
        self.vocabToNew = {}
        self.train_data, self.train_label = self.readFile(self.config.get_train_path())
        self.test_data, self.test_label = self.readFile(self.config.get_test_path())
        self.data = self.train_data + self.test_data
        self.label = self.train_label + self.test_label
        self.buildVocab()
        self.numVocab = len(self.w2vmodel.w2vmodel.vocab)

        # [self.testData, self.testLabel] =  self.readFile(testDataPath)
        self.weight = self.getW2V()

    def __len__(self):
        if self.aim == 'train':
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, item):
        if self.aim == 'train':
            w2v = []
            for each in self.train_data[item]:
                id = self.vocabToNew[each]
                w2v.append(id)
            while len(w2v) < 30:
                w2v.append(0)
            w2v = torch.Tensor(w2v).cuda().long() if self.config.cuda else torch.Tensor(w2v).long()
            # label = torch.zeros(len(self.cateVocab))
            # label[self.label[item] - 1] = 1
            # label = label.long().cuda()
            return w2v, self.train_label[item] - 1
        else:
            w2v = []
            for each in self.test_data[item]:
                id = self.vocabToNew[each]
                w2v.append(id)
            while len(w2v) < 30:
                w2v.append(0)
            w2v = torch.Tensor(w2v).cuda().long() if self.config.cuda else torch.Tensor(w2v).long()
            # label = torch.zeros(len(self.cateVocab))
            # label[self.label[item] - 1] = 1
            # label = label.long().cuda()
            return w2v, self.test_label[item] - 1

    def buildVocab(self):
        for eachV in self.train_data:
            for eachW in eachV:
                if eachW not in self.vocabToNew:
                    self.vocabToNew[eachW] = len(self.vocabToNew) + 1
                if eachW in self.w2vmodel.vocab:
                    self.vocabToOld[eachW] = self.w2vmodel.vocab[eachW].index
                else:
                    self.vocabToOld[eachW] = -1

        for eachV in self.test_data:
            for eachW in eachV:
                if eachW not in self.vocabToNew:
                    self.vocabToNew[eachW] = len(self.vocabToNew) + 1
                if eachW in self.w2vmodel.vocab:
                    self.vocabToOld[eachW] = self.w2vmodel.vocab[eachW].index
                else:
                    self.vocabToOld[eachW] = -1

        if '' not in self.vocabToNew:
            self.vocabToNew[''] = len(self.vocabToNew) + 1

    '''
    def buildTrainData(self):
        traindataloader = Reader.DataLoader(self.trainData, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.traindataiter = Reader.DataLoaderIter(traindataloader)

    def buildTestData(self):
        testdataloader = Reader.DataLoader(self.testData, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.testdataiter = Reader.DataLoaderIter(testdataloader)


    def next_batch(self,):
        if self.traindataiter is None:
            self.buildTrainData()
        try:
            batch = self.traindataiter.next()
            self.traindataiter += 1

            if self.is_cuda:
                batch = [batch[0].cuda(), batch[1].cuda(), batch[2].cuda()]
            return batch

        except StopIteration:  # 一个epoch结束后reload
            self.epoch += 1
            self.buildTrainData()
            self.iteration = 0  # reset and return the 1st batch

            batch = self.dataiter.next()
            if self.is_cuda:
                batch = [batch[0].cuda(), batch[1].cuda(), batch[2].cuda()]
            return batch
    '''
    '''
    def next_text_batch(self,):
        if self.testdataiter is None:
            self.buildTestData
        try:
            batch = self.dataiter.next()
            self.iteration += 1

            if self.is_cuda:
                batch = [batch[0].cuda(), batch[1].cuda(), batch[2].cuda()]
            return batch

        except StopIteration:  # 一个epoch结束后reload
            self.epoch += 1
            self.build()
            batch = self.dataiter.next()
            if self.is_cuda:
                batch = [batch[0].cuda(), batch[1].cuda(), batch[2].cuda()]
            self.iteration = 1  # reset and return the 1st batch
            return batch
    '''

    def getW2V(self):
        # padding is zero
        weight = torch.zeros(len(self.vocabToNew) + 1, self.embDim)
        for each in self.vocabToNew:
            if each in self.w2vmodel.w2vmodel.vocab:
                weight[self.vocabToNew[each], :] = torch.from_numpy(self.w2vmodel.w2vmodel[each])
                # weight[self.vocabToNew[each], :] = torch.from_numpy(self.w2vmodel[each])
        return weight

    def findEntity(self, origin, content):
        pos = []
        contentTemp = re.split('[ #,|]', content)
        for i in range(0, len(contentTemp)):
            if contentTemp[i] == 'G':
                pos.append(i)
        data = [origin.split(' ')[int(contentTemp[each - 2]):int(contentTemp[each - 1])] for each in pos]
        label = [contentTemp[each + 1] for each in pos]

        # for each in data:
        #    if len(each) > self.maxLen:
        #        self.maxLen = len(each)

        for i in range(0, len(label)):
            if not label[i] in self.cateVocab:
                self.cateVocab[label[i]] = len(self.cateVocab) + 1
            label[i] = self.cateVocab[label[i]]
        return [data, label]

    def readFile(self, path):
        try:
            text = open(path, "r", encoding="gbk").read()
        except:
            text = open(path, "r", encoding="utf-8").read()

        dataSet = []
        label = []

        data_list = text.split("\n\n")
        for data_index in range(len(data_list)):
            temp = data_list[data_index].split("\n")
            sentence_info = [item.split(" ") for item in temp]
            words = [sentence_info[i][0] for i in range(len(sentence_info))]

            for each_word in words:
                if each_word == '':
                    continue
                else:
                    if each_word not in self.vocabToNew:
                        self.vocabToNew[each_word] = len(self.vocabToNew) + 1

        # dataSet = [each.split() for each in dataSet]
        return dataSet, label

    def idx2setence(self, ids):
        return ''.join([self.w2vmodel.index2word[id - 2] for id in ids])


class w2VModel:
    def __init__(self, path, binary_wv_model=True):
        # self.w2vmodel = gensim.models.KeyedVectors.load_word2vec_format(path, binary=binary_wv_model, limit=10)
        self.w2vmodel = gensim.models.KeyedVectors.load_word2vec_format(path, binary=binary_wv_model)
