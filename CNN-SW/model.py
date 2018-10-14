# -*- coding:utf-8 -*-
#
# Created by ETIP-team
#
# On 2018-09-04

import re
import os
import jieba
import random
import shutil
import pynlpir   # license could be expired.
import numpy as np
import tensorflow as tf

import xml.etree.ElementTree as ET
from copy import deepcopy
from gensim.models import Word2Vec
import config as cfg

ignore_str = "，。,；：:…． "
splitChars = '[，。,；：:…．;]'

word_segmentation_method = cfg.WORD_SEGMENTATION_METHOD
if cfg.WORD_SEGMENTATION_METHOD == "jieba":
    word_vector_model = Word2Vec.load(cfg.JIEBA_WV_MODEL)
    jieba.load_userdict(cfg.JIEBA_USER_DICT)
elif cfg.WORD_SEGMENTATION_METHOD == "nlpir":
    word_vector_model = Word2Vec.load(cfg.NLPIR_WV_MODEL)
    pynlpir.open()
else:
    raise KeyError("Word segmentation method was invalid")

# start save from max_cross_entropy for every 0.01 less util min_cross_entropy

ls_cross_entropy_th = cfg.LS_CROSS_ENTROPY_TH

word_embedding_dim = cfg.WORD_EMBEDDING_DIM
sentence_length = cfg.SENTENCE_LENGTH
feature_maps_number = cfg.FEATURE_MAPS_NUM

kernal_length = cfg.KERNAL_LENGTH
beta = cfg.L2_BETA
th_c = cfg.TH_C
th_s = cfg.TH_S
th_iou = cfg.TH_IOU
min_len_th = cfg.MIN_LENGTH_TH
min_voter_number = cfg.MIN_VOTER_NUM
seedValue = cfg.SEED_VALUE
poolingOut = cfg.POOLING_OUT
poolingHeight = cfg.POOLING_HEIGHT
empty_wv = np.zeros(word_embedding_dim).reshape(1, word_embedding_dim)

label = cfg.LABEL

classes_num = cfg.CLASSES_NUM


class CNNTextClassifier(object):
    """A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, over-time-pooling and softmax layer."""
    def __init__(self, sentence_length=sentence_length, classes_num=classes_num,
                 feature_maps_number=feature_maps_number, l2_beta = beta, kernal_length=kernal_length):
        self.x = tf.placeholder(tf.float32, [None, sentence_length * word_embedding_dim], name="x")
        self.y_ = tf.placeholder(tf.float32, [None, classes_num], name="y_")
        self.input_x = tf.reshape(self.x, [-1, sentence_length, word_embedding_dim, 1], name="input_x")

        self.W_conv1 = weight_variable([kernal_length, word_embedding_dim, 1, feature_maps_number], name="W_conv1")
        self.b_conv1 = bias_variable([feature_maps_number], name="b_conv1")
        self.f_conv1 = tf.nn.relu(conv2d(self.input_x, self.W_conv1) + self.b_conv1, name="f_conv1")
        self.f_pool1 = tf.reshape(max_overtime_pooling(self.f_conv1), [-1, feature_maps_number*poolingOut], name="f_pool1")

        self.W_fc1 = weight_variable([feature_maps_number*poolingOut, classes_num], name="W_fc1")
        self.b_fc1 = bias_variable([classes_num], name="b_fc1")
        self.y_conv = tf.nn.softmax(tf.matmul(self.f_pool1, self.W_fc1) + self.b_fc1, name="y_conv")

        self.regularizer = tf.nn.l2_loss(self.W_fc1) + tf.nn.l2_loss(self.W_conv1, name="regularizer")

        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_conv), reduction_indices=[1]), name="cross_entropy_before_l2")
        self.cross_entropy = tf.add(self.cross_entropy, l2_beta * self.regularizer, name="cross_entropy_after_l2")

        self.train_step = tf.train.AdamOptimizer(1e-4, epsilon=1e-8).minimize(self.cross_entropy, name="train_step")

        self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1), name="correct_prediction")
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name="accuracy")


def weight_variable(shape, name):  # define weight initialize function
    initial = tf.truncated_normal(shape, stddev=0.1, seed=seedValue, name=name)
    return tf.Variable(initial)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_overtime_pooling(x):
    return tf.nn.max_pool(x, ksize=[1, poolingHeight, 1, 1], strides=[1, poolingHeight, 1, 1], padding='VALID')


def create_train_samples_batch(train_folder_path):
    ls_all_samples = []
    empty_ls = []
    for i in range(classes_num):
        ls_all_samples.append(empty_ls.copy())
    for file in os.listdir(train_folder_path):
        try:
            text = open(train_folder_path + file, "r").read()
            root = ET.fromstring(text)
            for child in root:
                for i in range(classes_num-2):
                    for sentence in child.findall(label[i]):
                        if sentence.text != None:
                            ls_all_samples[i].append(re.sub('\s', '', sentence.text))
                for sub_child_i in child.findall("情形"):
                    for sub_child_j in sub_child_i.findall(label[-2]):
                        if sub_child_j.text!= None:
                            ls_all_samples[-2].append(re.sub('\s', '', sub_child_j.text))
                    for sub_child_j in sub_child_i.findall(label[-1]):
                        if sub_child_j.text!= None:
                            ls_all_samples[-1].append(re.sub('\s', '', sub_child_j.text))
        except:
            print(file+"file read error!\n\n\n\n\n\n")
    all_count = 0
    for i in range(classes_num):
        count = len(ls_all_samples[i])
        print("label: "+label[i]+" sample numbers are {}".format(count))
        all_count += count

    print("{} training samples are loaded from {}".format(all_count, train_folder_path))

    return ls_all_samples


def get_train_batch(ls_all_samples):
    min_len = 999
    for i in range(classes_num):
        if len(ls_all_samples[i]) < min_len:
            min_len = len(ls_all_samples[i])
    ls_train = []
    min_len = min(10, min_len)
    for i in range(classes_num):
        one_hot_label = [0] * classes_num
        one_hot_label[i] = 1
        chosen_samples = random.sample(ls_all_samples[i], min_len)
        ls_train += list2wv(list_text_seg(chosen_samples.copy()), np.array(one_hot_label.copy()).reshape(1, classes_num))
    return ls_train


def cnn_batch_train(CNN, ls_train, train_times, entropy):
    # print(ls_train[0].shape, end="\n\n\n")
    # print(ls_train[1].shape)
    cross_entropy = CNN.cross_entropy.eval(feed_dict={CNN.x: ls_train[0], CNN.y_: ls_train[1]})

    if cross_entropy > entropy:
        # cross_entropy = CNN.cross_entropy.eval(feed_dict={CNN.x: ls_train[0], CNN.y_: ls_train[1]})
        if train_times % 100 == 0:
            train_accuracy = CNN.accuracy.eval(feed_dict={CNN.x: ls_train[0], CNN.y_: ls_train[1]})
            print("step {}, training accuracy {:f}".format(train_times, train_accuracy))
            print("step {}, cross_entropy {:f}\n".format(train_times, cross_entropy))
        CNN.train_step.run(feed_dict={CNN.x: ls_train[0], CNN.y_: ls_train[1]})
        return True
    else:
        cross_entropy = CNN.cross_entropy.eval(feed_dict={CNN.x: ls_train[0], CNN.y_: ls_train[1]})
        print("\ntrain finished, total step {}".format(train_times))
        print("finally cross_entropy {:f}\n".format(cross_entropy))
        return False


def list_text_seg(ls):
    """input the list of strings you want to cut by jieba
        return the list, every item contains the one list of original string"""
    for i in range(len(ls)):
        ls1 = []
        x = jieba.cut(ls[i])
        for a in x:
    #        if a != ' ':
           ls1.append(a)
        ls[i] = ls1
    return ls


def list2wv(ls, label=np.zeros((1, classes_num))):
    """Input: ls: the list of sub_lists, which contains the segment result of one string
                    label: one hot label.
         Return list of tuples, which numpy size is (1, sentence_length*word_vector_length)
         Note that sentence can be too long for the model"""
    ls_result = []
    for i in ls:
        arr = empty_wv
        for j in [item for item in i if item not in ignore_str]:
            try:
                arr = np.vstack((arr, word_vector_model.wv[j]))
            except KeyError:
                arr = np.vstack((arr, empty_wv))
                # print(j, "can not be found in word2vec dictionary")
        arr = np.delete(arr, 0, axis=0)
        row, column = arr.shape
        try:
            arr = np.pad(arr, ((0, sentence_length - row), (0, 0)), 'constant', constant_values=0).reshape(1,
                                                                                                       sentence_length*word_embedding_dim)
        except:
            print(list2str(i))
            exit("Above sentence length is"+str(len([item for item in i if item not in ignore_str]))+"was too long")
        ls_result.append((arr, label))

    return ls_result


def training_set(ls):
    """Input the list of tuples and let the (training_input, label) vstack into one batch
        return one tuple (training_batch, label_batch)"""
    train_set_input, train_set_label = ls[0]
    for i, j in ls:
        train_set_input = np.vstack((train_set_input, i))
        train_set_label = np.vstack((train_set_label, j))
    train_set_input = np.delete(train_set_input, 0, axis=0)
    train_set_label = np.delete(train_set_label, 0, axis=0)
    return train_set_input, train_set_label


def paragraph_detection(text, sess, th_c=th_c):
    """The work flow of text detection.
        All steps with the structure:
        list of structure which is [[w1,w2,w3], softmax_result, position_tag]"""
    # cut the sentence to word segmentation.
    ls_paragraph_detection_result = []
    for sub_list in txt_split(text):
        ls_sliding_windows_text = []
        for i in range(1, min(sentence_length, len(sub_list))+1):
            for j in range(0, min(sentence_length, len(sub_list)) - i + 1):
              ls_sliding_windows_text.append((sub_list[j:j + i], [pos_tag for pos_tag in range(j, j + i)]))

        # # Work flow
        # 1 Get candidate windows by threshold
        ls_sliding_window_candidate = detection_sliding_window_text(ls_sliding_windows_text, sess, th_c)
        # 2 Link phrases with the same label among nighboring sentences
        ls_concatenation = same_label_concatenation(deepcopy(ls_sliding_window_candidate))
        # 3 Delete duplicated phrase with same label but different score
        ls_concatenation = delete_duplicated_phrase(deepcopy(ls_concatenation))
        # 4 Judge the overlapping sentences in different labels by voting
        ls_label_conflict_voting = classification_conflict_resolving(deepcopy(ls_concatenation), deepcopy(ls_sliding_window_candidate))

        ls_paragraph_detection_result += ls_label_conflict_voting

        # Only ls_senetence_window_candidate
        # ls_sliding_window_candidate = detection_sliding_window_text(ls_sliding_windows_text, sess, th_c)
        # ls_paragraph_detection_result += ls_sliding_window_candidate

    return ls_paragraph_detection_result


def text_match(main_list, sub_list):
    """Check if the sub_list in main_list """
    for main_index in range(len(main_list)-len(sub_list)+1):
        if main_list[main_index:main_index+len(sub_list)] == sub_list:
            return True
    return False


def neighboring_merge(left_list, right_list):
    """Combine two sentences which are part duplicated
        For instance:
        [w1, w2, w3] and [w3, w4, w5, w6]
        Return [w1, w2, w3, w4, w5, w6] """
    min_length = min(len(left_list[2]), len(right_list[2]))
    for index in range(-1, -min_length - 1, -1):
        if left_list[2][index:] == right_list[2][:abs(index)]:
            return left_list[0][:] + right_list[0][abs(index):]
    return []


def list2str(ls):
    return "".join(ls)


def detection_sliding_window_text(ls_sliding_windows_text, sess, th_c):
    """Input the list of the words combination, pre-trained CNN(tensorflow session) to detect
        Return  the list of sub_lists which format is [words_combination, softmax_result, position_tag]"""
    ls_sliding_window_candidate = []
    for words_combination, position_tag in ls_sliding_windows_text:
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y_ = graph.get_tensor_by_name("y_:0")
        softmax_result = sess.run("y_conv:0", feed_dict={x: list2wv([words_combination])[0][0], y_: np.zeros((1, classes_num))}).reshape(classes_num,)
        if max(softmax_result) > th_c:
            ls_sliding_window_candidate.append([words_combination, softmax_result, position_tag])
    return ls_sliding_window_candidate


def delete_duplicated_phrase(ls_concatenation):
    """Input:  list of phrase after concatenation
        Return: list of phrase with less concatenation result
        After concatenation, the word segmentation combination could be same with different softmax score
        Select the maximum for one phrase and drop the other.
        """
    ls_not_remove = [True] * len(ls_concatenation)
    for i in range(len(ls_concatenation)):
        if ls_not_remove[i] is False:
            continue
        for j in range(len(ls_concatenation)):
            if ls_not_remove[j] is False:
                    continue
            if i != j:
                if np.argmax(ls_concatenation[i][1]) == np.argmax(ls_concatenation[j][1]):  # label same
                    if set(ls_concatenation[i][2]) >= set(ls_concatenation[j][2]):
                        if label[np.argmax(ls_concatenation[i][1])] == "保障范围":
                             continue
                        if set(ls_concatenation[i][2]) == set(ls_concatenation[j][2]):
                            if max(ls_concatenation[i][1]) > max(ls_concatenation[j][1]):
                                ls_concatenation[i][1] = ls_concatenation[j][1]
                        ls_not_remove[j] = False

    return [ls_concatenation[i] for i in range(len(ls_concatenation)) if ls_not_remove[i] is True]


def same_label_concatenation(ls_sliding_window_candidate):
    """Input: list of sliding_window_candidate
         Return: list of concatenation phrase
         concatenation the same label phrase."""
    ls_not_remove = [True] * len(ls_sliding_window_candidate)  # set the tag of the detection result
    for i in range(len(ls_sliding_window_candidate)):
            if ls_not_remove[i] is True:
                for j in range(len(ls_sliding_window_candidate)):
                    if i != j:
                        if np.argmax(ls_sliding_window_candidate[i][1]) == np.argmax(ls_sliding_window_candidate[j][1]):
                            if ls_not_remove[j] is True:
                                overlap_len = len(set(ls_sliding_window_candidate[i][2]) & set(ls_sliding_window_candidate[j][2]))
                                min_len = min(len(set(ls_sliding_window_candidate[i][2])), len(set(ls_sliding_window_candidate[j][2])))
                                if overlap_len > 0 and overlap_len != min_len:
                                    if len(neighboring_merge(ls_sliding_window_candidate[i], ls_sliding_window_candidate[j])):
                                        ls_sliding_window_candidate[i][0] \
                                            = neighboring_merge(ls_sliding_window_candidate[i], ls_sliding_window_candidate[j])
                                    else:
                                        ls_sliding_window_candidate[i][0] \
                                             = neighboring_merge(ls_sliding_window_candidate[j], ls_sliding_window_candidate[i])
                                    ls_not_remove[j] = False
                                    ls_sliding_window_candidate[i][2] = list(
                                    set(ls_sliding_window_candidate[i][2]) | set(ls_sliding_window_candidate[j][2]))
                                    ls_sliding_window_candidate[i][1] \
                                    = (ls_sliding_window_candidate[i][1] + ls_sliding_window_candidate[j][1]) / 2
                                # if neighboring {1,2,3}  and {4,5,6} then union two sets
                                elif max(ls_sliding_window_candidate[i][2])+1 == min(ls_sliding_window_candidate[j][2]):
                                        ls_sliding_window_candidate[j][2] = list(set(ls_sliding_window_candidate[i][2]) | set(ls_sliding_window_candidate[j][2]))
                                        ls_sliding_window_candidate[j][0] = ls_sliding_window_candidate[i][0]+ls_sliding_window_candidate[j][0]
                                        ls_not_remove[i] = False
                                elif max(ls_sliding_window_candidate[j][2])+1 == min(ls_sliding_window_candidate[i][2]):
                                        ls_sliding_window_candidate[j][2] = list(set(ls_sliding_window_candidate[i][2]) | set(ls_sliding_window_candidate[j][2]))
                                        ls_sliding_window_candidate[j][0] = ls_sliding_window_candidate[j][0]+ls_sliding_window_candidate[i][0]
                                        ls_not_remove[i] = False
    return [ls_sliding_window_candidate[i] for i in range(len(ls_sliding_window_candidate)) if ls_not_remove[i] is True]


# Detect coincident labels
def classification_conflict_resolving(ls_phrase_concatenation, ls_sliding_window_candidate):
    """Input: ls_phrase_concatenation: list of phrase after concatenation
                    ls_sliding_window_candidate:   list of candidate after sliding window and CNN scoring
        Return: list of phrase after voting.
        Voting method to solve classification conflict"""
    ls_not_remove = [True] * len(ls_phrase_concatenation)
    for i in range(len(ls_phrase_concatenation)):
        if ls_not_remove[i] == False:
            continue
        for j in range(len(ls_phrase_concatenation)):
            if ls_not_remove[j] == False:
                continue
            if i != j:
                if np.argmax(ls_phrase_concatenation[i][1]) != np.argmax(ls_phrase_concatenation[j][1]):
                    if max(ls_phrase_concatenation[i][1]) > th_s and max(ls_phrase_concatenation[j][1]) > th_s:
                        continue
                    else:
                        intersection_length = len(
                            set(ls_phrase_concatenation[i][2]) & set(ls_phrase_concatenation[j][2]))
                        union_length = len(
                            set(ls_phrase_concatenation[i][2]) | set(ls_phrase_concatenation[j][2]))
                        min_len = min(len(ls_phrase_concatenation[i][2]), len(ls_phrase_concatenation[j][2]))
                        if intersection_length / union_length >= th_iou or min_len >= min_len_th:
                            vi = vj = 0
                            for n in range(len(ls_sliding_window_candidate)):
                                if set(ls_sliding_window_candidate[n][2]) < set(ls_phrase_concatenation[i][2]) \
                                    and np.argmax(ls_sliding_window_candidate[n][1]) == np.argmax(ls_phrase_concatenation[i][1]):
                                        vi = vi+1
                                if set(ls_sliding_window_candidate[n][2]) < set(ls_phrase_concatenation[j][2]) \
                                    and np.argmax(ls_sliding_window_candidate[n][1]) == np.argmax(ls_phrase_concatenation[j][1]):
                                        vj = vj+1
                            if vi <= min_voter_number and vj <= min_voter_number:
                                  # the number of voter is not enough
                                if max(ls_phrase_concatenation[i][1]) > max(ls_phrase_concatenation[j][1]):
                                    ls_not_remove[j] = False
                                    continue
                                else:
                                    ls_not_remove[i] = False
                                    continue
                            if vi > vj:
                                ls_not_remove[j] = False
                            if vi < vj:
                                ls_not_remove[i] = False
                            if vi == vj:
                                 if max(ls_phrase_concatenation[i][1]) > max(ls_phrase_concatenation[j][1]):
                                    ls_not_remove[j] = False
                                 else:
                                    ls_not_remove[i] = False

    return [ls_phrase_concatenation[index] for index in range(len(ls_phrase_concatenation)) if ls_not_remove[index] is True]


def txt_split(text_str):
    """Input: one paragraph.
        Return: sentence word cut list.
        like: [["w1", "w2", "w3"], ["w1", "w2"]].
        For each one sentence in this paragraph, we return the word segmentation of one sentence.
    """
    ls_seg_sentences = []
    if word_segmentation_method == "jieba":
        for item in [sentence for sentence in re.split(splitChars, text_str) if len(sentence) > 0]:
            ls_seg_sentences.append([seg for seg in jieba.cut(item)])
    elif word_segmentation_method == "nlpir":
        for item in [sentence for sentence in re.split(splitChars, text_str) if len(sentence) > 0]:
            ls_seg_sentences.append([seg for seg, flag in pynlpir.segment(item)])
    else:
        raise KeyError("Word segmentation method was invalid")
    return ls_seg_sentences


def sentence_in_paragraph_match(ls_paragraph_detection_result):
    """Input: detection result for whole paragraph
         Return delete duplicated detection result"""
    ls_contained_tag = [True] * len(ls_paragraph_detection_result)
    for i in range(len(ls_paragraph_detection_result)):
        if ls_contained_tag[i] is False:
            continue
        for j in range(len(ls_paragraph_detection_result)):
            if ls_contained_tag[j] is False:
                continue
            if i != j and np.argmax(ls_paragraph_detection_result[i][1]) == np.argmax(ls_paragraph_detection_result[j][1]):
                if list2str(ls_paragraph_detection_result[i][0]).find(list2str(ls_paragraph_detection_result[j][0])) > -1:
                    if max(ls_paragraph_detection_result[i][1]) > max(ls_paragraph_detection_result[j][1]):
                        ls_contained_tag[j] = False
                    else:
                        ls_contained_tag[i] = False
    ls_sentence_in_paragraph_match_result = [ls_paragraph_detection_result[index] for index in range(len(ls_paragraph_detection_result)) if ls_contained_tag[index] is True]
    return ls_sentence_in_paragraph_match_result


def write2file(ls_sentence_result, write_file_opened):
    """Write all detection result to contract except CP and IA(will combine to condition in real contract)."""
    for item in ls_sentence_result:
        if np.argmax(item[1]) in range(classes_num-2):
            # write_file_opened.write(list2str(item[0]).lstrip(remove_chars)+" ({:.3f}) \n".format(max(item[1])))
            write_file_opened.write("<" + label[np.argmax(item[1])] + ">")
            write_file_opened.write(list2str(item[0]))
            write_file_opened.write("</" + label[np.argmax(item[1])] + ">\n")


def write_condition2file(ls_sentence_result, write_file_opened):
    """In real contract, the arbitrary number condition for the payment (CP) and any number of insurance amount(IA)
        combine to one condition in logic."""
    i = 0
    situation = 1
    # ls_sentence_result = [item for item in ls_sentence_result if label[np.argmax(item[1])] != "保障范围"]
    ls_sentence_result = [item for item in ls_sentence_result if np.argmax(item[1]) not in range(classes_num-2)]
    while(i < len(ls_sentence_result)):
        write_file_opened.write("<情形>\n")
        condition = 1
        money = 1
        while(i < len(ls_sentence_result) and label[np.argmax(ls_sentence_result[i][1])] == "给付条件"):
            write_file_opened.write("<给付条件>")
            # print("<给付条件{}>".format(condition))
            # write_file_opened.write("\t\t\t" + list2str(one_result[0]) + " ({:.3f})\n".format(max(one_result[1])))
            write_file_opened.write(list2str(ls_sentence_result[i][0]) + "</给付条件>\n")
            # print(list2str(ls_sentence_result[i][0]) + "</给付条件{}>\n".format(condition))
            condition = condition + 1
            i += 1
        while(i < len(ls_sentence_result) and label[np.argmax(ls_sentence_result[i][1])] == "给付金额"):
            write_file_opened.write("<给付金额>")
            write_file_opened.write(list2str(ls_sentence_result[i][0]) + "</给付金额>\n")
            money = money + 1
            i += 1
        write_file_opened.write("</情形>\n")
        situation += 1


def detect_file_and_write(detect_file_path, write_file_path, sess):
    contractText = open(detect_file_path, "r").read()
    # if not os.path.exists("".join(write_file_path.split("/")[:-1])):
    #     os.makedirs("".join(write_file_path.split("/")[:-1]))
    write_file = open(write_file_path, "w")
    if contractText == None:
        # continue
        print("\n\n Wrong file!" + detect_file_path)
        exit()
    try:
        root = ET.fromstring(contractText)
    except:
        print("\n\n Wrong file!" + detect_file_path)
        exit()
    write_file.write("<合同>\n")
    for insurance in [child for child in root]:
        write_file.write("<" + insurance.tag + ">\n")
        original = insurance.find("原文")
        text = original.text
        text = re.sub('\s', '', text)
        write_file.write("<原文> {}</原文>\n".format(text))
        ls_paragraph_detection_result = paragraph_detection(text, sess, th_c)
        ls_in_paragraph_match_result = sentence_in_paragraph_match(ls_paragraph_detection_result)
        write2file(ls_in_paragraph_match_result, write_file)
        write_condition2file(ls_paragraph_detection_result, write_file)
        write_file.write("</" + insurance.tag + ">\n")
    write_file.write("</合同>")
    write_file.close()


def create_train_path(dataset_path, fold_index):
    """Create the temporary training set for this fold."""
    temp_train_path = "./train_data_path/"
    if not os.path.exists(temp_train_path):
        os.makedirs(temp_train_path)
    else:
        return temp_train_path
    fold_paths = os.listdir(dataset_path)
    for fold_i in range(cfg.K_FOLD):
        if fold_i == fold_index:
            continue
        else:
            one_fold_path = dataset_path+fold_paths[fold_i]+"/"
            for file_in_one_fold in os.listdir(one_fold_path):
                shutil.copyfile(one_fold_path+file_in_one_fold, temp_train_path+file_in_one_fold)
    return temp_train_path


def remove_train_path(train_data_path):
    """Remove the temporary train data path"""
    for file in os.listdir(train_data_path):
        os.remove(train_data_path+file)
    os.rmdir(train_data_path)

