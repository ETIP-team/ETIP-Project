# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2018-10-19

import os
import jieba
import re
import xml.etree.ElementTree as ET
from copy import deepcopy

jieba.load_userdict("jieba_userdict.txt")

files_folder_path = ""   # end with a /

write_files_folder_path = ""  # end with a /

split_chars = '[，。,；：:…．;]'

label = ['保险期间', '合同终止', '责任免除', '保障范围', '等待期', '给付条件', '给付金额']
ner_label = ['PC', 'T', 'E', 'C', 'WP', 'CP', 'IA']
label_number = len(label)


def empty_list():
    empty_list = []
    detection_list = []
    for i in range(label_number):
        detection_list.append(empty_list[:])
    return deepcopy(detection_list)


def find_position(label_sentence, cut_list):
    result_ls = []
    start_flag = False
    start_position = -1
    end_position = -1
    for word_index in range(len(cut_list)):
        word = cut_list[word_index]
        if label_sentence.find(word) > -1:
            if not start_flag :
                start_position = word_index
                end_position = word_index
                start_flag = True
            else:
                end_position = word_index
        else:
            if start_flag:
                start_flag = False
                result_ls.append((start_position, end_position+1))
                start_position = -1
                end_position = -1
    if start_flag:
        end_position = len(cut_list)
        result_ls.append((start_position, end_position))
    return result_ls


file_ls = os.listdir(files_folder_path)
for file in file_ls[:]:
    file_path = files_folder_path + file
    open_file = open(write_files_folder_path+file, "w")
    print("This is ", file_ls.index(file), " th file")
    contractText = open(file_path, "r").read()
    if contractText == None:
        exit("this file is empty!")
    root = ET.fromstring(contractText)
    for child in root:
        original_text = re.sub('\s', '', child.find("原文").text)
        original_sentecens = re.split(split_chars, original_text)
        detection_list = empty_list()
        for index in range(len(label[:-2])):
            for sentence in child.findall(label[index]):
                if sentence.text != None:
                    detection_list[index].append(re.sub('\s', '', sentence.text))
        for sub_child_i in child.findall("情形"):
            for sub_child_j in sub_child_i.findall(label[-2]):
                if sub_child_j.text != None:
                    detection_list[-2].append(re.sub('\s', '', sub_child_j.text))
            for sub_child_j in sub_child_i.findall(label[-1]):
                if sub_child_j.text != None:
                    detection_list[-1].append(re.sub('\s', '', sub_child_j.text))

        for sentence in re.split(split_chars, original_text):
            #     print(sentence)
            cut_list = [item for item in jieba.cut(sentence)]
            hit_flag = False
            hit_result = []
            for index in range(label_number):
                for label_sentence in detection_list[index]:
                    if sentence.find(label_sentence) > -1:
                        print("sentence", sentence)
                        print("label_sentence", label_sentence)
                        find_result = find_position(label_sentence, cut_list)
                        if len(find_result) > 0:
                            hit_flag = True
                        for one_result in find_result:
                            print("find position in ", one_result)
                            write_flag = input("Write done?\n")
                            if write_flag.lower() in ["yes", "y", ""]:
                                hit_result.append((ner_label[index], one_result))
                                print("Write!")
                            else:
                                print("Ignore!")
                                continue
            if hit_flag:
                for item in cut_list[:-1]:
                    open_file.write(item + " ")
                open_file.write(cut_list[-1] + "\n")
                for item in hit_result[:-1]:
                    open_file.write(str(item[1][0]) + "," + str(item[1][1]) + " G#" + item[0] + "|")
                open_file.write(str(item[1][0]) + "," + str(item[1][1]) + " G#" + item[0] + "\n\n")
            print("\n\nWrite Done!\n\n")
        # break


