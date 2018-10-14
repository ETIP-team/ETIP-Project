import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import config as cfg
from copy import deepcopy

th_p = cfg.TH_P
min_th_p = cfg.MIN_TH_P
label = cfg.LABEL
ls_cross_entropy_th = cfg.LS_CROSS_ENTROPY_TH
classes_num = cfg.CLASSES_NUM

metrics = ['Precision', 'Recall', 'F1_score', 'TPR', 'FPR']
reindex_ls = ['保障范围', '等待期', '保险期间', '给付条件', '给付金额', '责任免除', '合同终止']
reindex_simple = ['C', 'WP', 'PC', 'CP', 'IA', 'E', 'T', 'neg.']
empty_dir = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}


def neighboring_merge(left_list, right_list):
    """Compute the union and intersection of two sentences which can be part duplicated
        For instance:
        "w1w2w3" and "w3w4w5w6"
        Return "w1w2w3w4w5w6" and "w3" """
    min_length = min(len(left_list), len(right_list))
    current_index = 999
    for index in range(-1, -min_length - 1, -1):
        if left_list[index:] == right_list[:abs(index)]:
            current_index = index
    if current_index < 0:
        union = left_list + right_list[abs(current_index):]
        return union, left_list[current_index:]
    else:
        return "", ""


def intersection_over_union(detected_sentence, answer_sentence):
    """Input: str, str
        Return: float
        Compute the intersection over union of two sentences"""
    if detected_sentence.find(answer_sentence) != -1:
        return float(len(answer_sentence)/len(detected_sentence))
    elif answer_sentence.find(detected_sentence) != -1:
        return float(len(detected_sentence)/len(answer_sentence))
    else:
        (union1, intersection1) = neighboring_merge(detected_sentence, answer_sentence)
        (union2, intersection2) = neighboring_merge(answer_sentence, detected_sentence)
        if len(union1) > 0 and len(union2) > 0:
            return max(float(len(intersection1)/len(union1)), float(len(intersection2)/len(union2)))
        elif len(union1) > 0:
            return float(len(intersection1)/len(union1))
        elif len(union2) > 0:
            return float(len(intersection2)/len(union2))
        else:
            return 0.


def empty_list():
    empty_list = []
    detection_list = []
    for i in range(classes_num):
        detection_list.append(empty_list[:])
    return detection_list


def check_label(detected_sentence, answer_list, check_type):
    """Input:
            detected_sentence: [str, 0]
            answer_list: list of sublist which is [str, 0]
            check_type: str
        Return: boolean
        For different check type return if hit answer"""
    th_p_ = min_th_p if len(detected_sentence[0]) <= 8 else th_p
    for index in range(len(answer_list)):
        if intersection_over_union(detected_sentence[0], answer_list[index][0]) >= th_p_:
            if check_type == "TP":
                detected_sentence[1] += 1
                if answer_list[index][1] == 0:
                    answer_list[index][1] += 1
                    return True
            elif check_type == "FP":
                if detected_sentence[1] == 0:  # was matched by other sentence.
                    detected_sentence[1] += 1
                    return True
            elif check_type == "FN":
                if answer_list[index][1] == 0:
                    answer_list[index][1] += 1
                    return True
    return False


def reset_the_list(to_reset_list):
    """After detected for one label, reset the hit tag to 0"""
    for label_index in range(classes_num):
        for j in range(len(to_reset_list[label_index])):
            to_reset_list[label_index][j][1] = 0


def measure_the_metrics(detection_list, answer_list, confusion_matrix, ls_metrics_dir):
    """Check the sentence in detection list if hit in answer list or not and fill the confusion matrix"""
    for index in range(classes_num):
        for detected_sentence in detection_list[index]:
            check = check_label(detected_sentence, answer_list[index], "TP")
            if check:
                confusion_matrix[index, index] += 1
                ls_metrics_dir[index]['TP'] += 1
        for other_index in range(classes_num):
            if other_index == index:
                continue
            else:
                for detected_other_label_sentence in detection_list[other_index]:
                    check = check_label(detected_other_label_sentence, answer_list[index], "FN")
                    if check:
                        ls_metrics_dir[index]['FN'] += 1
                        confusion_matrix[index, other_index] += 1
                    else:
                        ls_metrics_dir[index]['TN'] += 1
                for detected_sentence in detection_list[index]:
                    check = check_label(detected_sentence, answer_list[index], "FP")
                    if check:
                        ls_metrics_dir[index]['FP'] += 1
                        confusion_matrix[other_index, index] += 1
        for i in range(len(answer_list[index])):
            if answer_list[index][i][1] == 0:
                ls_metrics_dir[index]['FN'] += 1
                confusion_matrix[index, classes_num] += 1
        for i in range(len(detection_list[index])):
            if detection_list[index][i][1] == 0:
                ls_metrics_dir[index]['FP'] += 1
                confusion_matrix[classes_num, index] += 1
        reset_the_list(answer_list)
        reset_the_list(detection_list)


def parser_xml_file_and_get_list(file_path):
    contractText = open(file_path).read()
    if contractText == None:
        exit("this file is empty!")
    try:
        root = ET.fromstring(contractText)
    except:
        print("\n\n Wrong file!" + file_path)
    result_list = []
    for child in root:
        detection_list = empty_list()
        for index in range(len(label[:-2])):
            for sentence in child.findall(label[index]):
                if sentence.text != None:
                    detection_list[index].append([re.sub('\s', '', sentence.text), 0])
        for sub_child_i in child.findall("情形"):
            for sub_child_j in sub_child_i.findall(label[-2]):
                if sub_child_j.text != None:
                    detection_list[-2].append([re.sub('\s', '', sub_child_j.text), 0])
            for sub_child_j in sub_child_i.findall(label[-1]):
                if sub_child_j.text != None:
                    detection_list[-1].append([re.sub('\s', '', sub_child_j.text), 0])
        result_list.append(detection_list[:])
    return deepcopy(result_list)


def compare_with_answer(detect_path, answer_path):
    ls_metrics_dir = []
    for i in range(classes_num):
        ls_metrics_dir.append(empty_dir.copy())

    ls_detect_folder = os.listdir(detect_path)
    ls_answer_folder = os.listdir(answer_path)

    confusion_matrix = np.zeros((classes_num + 1, classes_num + 1), dtype=np.int32)
    for index_k_folder in range(len(ls_detect_folder)):
        for file in os.listdir(detect_path + ls_detect_folder[index_k_folder] + "/"):
            contract_detection_list = parser_xml_file_and_get_list(detect_path + ls_detect_folder[index_k_folder] + "/" + file)
            contract_answer_list = parser_xml_file_and_get_list(answer_path + ls_answer_folder[index_k_folder] + "/" + file[:-4] + ".xml")
            for i in range(len(contract_detection_list)):
                measure_the_metrics(contract_detection_list[i], contract_answer_list[i], confusion_matrix, ls_metrics_dir)
    return ls_metrics_dir, confusion_matrix


def reindex_confusion_matrix(confusion_matrix):
    """input 8*8 with neg.
        Reindex the label  (For paper)"""
    classes_simple = ['PC', 'T', 'E', 'C', 'WP', 'CP', 'IA', 'neg.']
    every_tag_dir = {}
    for one_label in classes_simple:
        every_tag_dir[one_label] = {}
        for one2label in classes_simple:
            every_tag_dir[one_label][one2label] = 0

    new_confusion_matrix = confusion_matrix.copy()
    for i in range(len(classes_simple)):
        for j in range(len(classes_simple)):
            every_tag_dir[classes_simple[i]][classes_simple[j]] = confusion_matrix[i][j]
            new_confusion_matrix[i][j] = 0

    for i in range(len(reindex_simple)):
        for j in range(len(reindex_simple)):
            new_confusion_matrix[i][j] = every_tag_dir[reindex_simple[i]][reindex_simple[j]]

    return new_confusion_matrix


def get_count_dataframe_by_confusion_matrix(confusion_matrix):
    experiment_metrics = []
    sum_result_dir = {"TP": 0, "FP": 0, "FN": 0}
    for one_label in range(classes_num):
        TP = confusion_matrix[one_label][one_label]
        sum_result_dir["TP"] += TP
        FP = 0
        for column_index in range(classes_num+1):
            FP += confusion_matrix[column_index][one_label]
        FP -= TP
        sum_result_dir["FP"] += FP
        FN = sum(confusion_matrix[one_label]) - TP

        sum_result_dir["FN"] += FN
        precision = TP/(TP+FP) if TP + FP != 0 else 0
        recall = TP/(TP+FN) if TP+FN != 0 else 0
        F1_score = (2*precision*recall)/(precision+recall) if precision*recall != 0 else 0
        experiment_metrics.append([precision, recall, F1_score])

    return pd.DataFrame(experiment_metrics, columns=metrics[:-2], index=label), sum_result_dir


def main():
    test_path = cfg.TEST_PATH
    print(test_path.split("/")[-2])
    answer_path = "data/"
    max_score = 0
    index = -1
    best_performance_data_frame = None
    best_confusion_matrix = []
    best_sum_result = {}
    best_ls_metrics_dir = []
    for entropy_index in range(min(len(os.listdir(test_path)), len(ls_cross_entropy_th))):
        str_entropy = str(ls_cross_entropy_th[entropy_index])
        if len(str_entropy) < len(str(0.12)):
            str_entropy += "0"
        str_entropy = "cross_entropy" +str_entropy +"/"
        detect_path = test_path + str_entropy
        ls_metrics_dir, confusion_matrix = compare_with_answer(detect_path, answer_path)
        # use confusion matrix to compute the result.
        data_frame, sum_result_dir = get_count_dataframe_by_confusion_matrix(confusion_matrix)
        score = data_frame['F1_score'].mean()
        if max_score < score:
            max_score = score
            index = entropy_index
            best_performance_data_frame = data_frame
            best_confusion_matrix = confusion_matrix
            best_sum_result = sum_result_dir
            best_ls_metrics_dir = ls_metrics_dir

    print("cross entropy:\n", ls_cross_entropy_th[index])
    best_performance_data_frame = best_performance_data_frame.reindex(reindex_ls)
    print("best performance:\n", best_performance_data_frame[metrics[:-2]])

    # for item in best_ls_metrics_dir:
    #     print(item)

    precision = best_sum_result["TP"]/(best_sum_result["TP"]+best_sum_result["FP"])
    recall = best_sum_result["TP"]/(best_sum_result["TP"]+best_sum_result["FN"])
    print("Overall:")
    print("Precision  ", precision)
    print("Recall  ", recall)
    print("F1 score  ", (2*precision*recall)/(precision+recall))


if __name__ == '__main__':
    main()
