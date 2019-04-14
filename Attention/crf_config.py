# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2019-04-03

import os
import torch as t


class Config:
    def __init__(self):
        # global config
        self.cuda = True  # False
        self.WORD_VEC_MODEL_PATH = "./model/word_vector_model/wikipedia-pubmed-and-PMC-w2v.bin"  # ACE05
        # self.WORD_VEC_MODEL_PATH = "./model/word_vector_model/bio_nlp_vec.tar/bio_nlp_vec/PubMed-shuffle-win-2.bin"
        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"

        # model config.

        self.attention_method = "general"  # "general",  "dot",  "concate"
        self.embedding_dim = 200
        # self.num_embeddings = 5
        self.hidden_units = 100  # embedding size must equals hidden_units * 2
        self.linear_hidden_units = 150  # 70
        self.encode_num_layers = 1
        self.decode_num_layers = 1
        self.encode_bi_flag = True

        self.learning_rate = 3e-4
        self.l2_penalty = 1e-4

        self.dataset_type = "ACE05"
        self.labels = ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA']
        self.bio_labels = ["O"]
        for one_label in self.labels:
            self.bio_labels.extend(["B-" + one_label, "I-" + one_label])

        self.bio_labels.extend([self.START_TAG, self.STOP_TAG])
        self.tag_to_ix = {}
        for i, tag in enumerate(self.bio_labels):
            self.tag_to_ix[tag] = i
        self.classes_num = len(self.bio_labels)  # Begin, Inside, Out of entity
        self.max_nested_level = 1

        # train config
        self.num_batch = 4

        self.max_epoch = 20
        self.start_save_epoch = 1

        self.start_test_epoch = 1

        self.train_data = None
        self.train_label = None
        self.train_str = None
        self.dev_data = None
        self.dev_label = None

        self.test_data = None
        self.test_str = None
        self.test_label = None
        self.metric_dicts = None

        self.output_path = "./result/result.data"

    def model_save_path(self, epoch, create_flag=True):
        final_model_path = "./model/" + self.dataset_type + "/"
        final_model_path += "crf_bi_" if self.encode_bi_flag else ""
        final_model_path += self.attention_method
        final_model_path += "_max_nested_level_" + str(self.max_nested_level)
        final_model_path += "_hidden_units_" + str(self.hidden_units)
        final_model_path += "_learning_rate_" + str(self.learning_rate)
        final_model_path += "_num_batch_" + str(self.num_batch)
        final_model_path += "_l2_" + str(self.l2_penalty)
        final_model_path += "_1_linear"
        # todo add.

        final_model_path += "/"
        if create_flag and not os.path.exists(final_model_path):
            os.makedirs(final_model_path)
            print("create model dir " + final_model_path + " successfully")

        return final_model_path + "model_epoch_" + str(epoch + 1) + ".pth"

    def load_model(self, model, epoch):
        model_path = self.model_save_path(epoch, False)
        model.load_state_dict(t.load(model_path))

        print("load model from " + model_path)
        return model

    def save_model(self, model, epoch):
        model_path = self.model_save_path(epoch, True)

        t.save(model.state_dict(), model_path)

        print("model saved in " + model_path + " successfully")
        return

    def get_train_path(self):
        return "./data/big_first/layer_train.data"

    def get_dev_path(self):
        return "./data/big_first/layer_dev.data"

    def get_test_path(self):
        return "./data/big_first/layer_test.data"


if __name__ == '__main__':
    config = Config()
    config.model_save_path(-1)
