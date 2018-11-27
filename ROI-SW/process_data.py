# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2018-10-08


import os
import pickle
import numpy as np
from utils import calc_ious_1d, bbox_transform_1d
import config as cfg

# th_iou_train = cfg.TH_IOU_TRAIN

all_mean = []
all_deviation = []
nega_num = [0]
gap_th = 0.2
nega_length_th = 3

cover_nega = open("global_cover_nega.txt", "w")


def feature_scaling(ndarray_all, pos_indexes, add_flag=False):
    mean_ls = []
    deviation_ls = []
    result_ls = []
    norm_ndarray = ndarray_all[pos_indexes]
    after_norm_ndarray = ndarray_all.copy()
    for i in range(norm_ndarray.shape[1]):
        column = norm_ndarray[:, i]
        mean = np.mean(column)
        deviation = np.std(column, ddof=1)
        deviation_ls.append(deviation)
        mean_ls.append(mean)
        if deviation != 0:
            result_ls.append(((column - mean) / deviation))
        else:
            result_ls.append((column - mean))
    if add_flag:
        all_mean.append(mean_ls)
        all_deviation.append(deviation_ls)
    after_norm_ndarray[pos_indexes] = np.array(np.mat(result_ls).T)
    return after_norm_ndarray


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


def get_all_gt(all_data):
    sample_num = len(all_data)
    all_gt_ls = [[] for i in range(cfg.CLASSES_NUM + 1)]
    for sentence_index in range(sample_num):
        bbox = all_data[sentence_index]["ground_truth_bbox"]
        cls = all_data[sentence_index]["ground_truth_cls"]
        gt_str_ls = all_data[sentence_index]["str"].split(" ")
        for sample_index in range(len(cls)):
            sample = bbox[sample_index]
            all_gt_ls[cls[sample_index]].append(gt_str_ls[sample[0]: sample[1] + 1])

    return all_gt_ls


def check_global_nega(all_gt_ls, bbox, sentence_str, th_iou_train):
    wait = True
    sentence_str_ls = sentence_str.split(" ")[bbox[0]: bbox[1] + 1]
    max_iou = 0
    for cls_index in range(cfg.CLASSES_NUM + 1):
        for str_ls in all_gt_ls[cls_index]:
            # if "".join(sentence_str_ls).find("".join(str_ls)) > -1:
            #     return False
            intersection = len(set(str_ls) & set(sentence_str_ls))
            union = len(set(str_ls) | set(sentence_str_ls))
            iou = intersection / union
            if union - intersection <= nega_length_th:
                return False
            if iou > max_iou:
                max_iou = iou
    if max_iou <= th_iou_train - gap_th:
        return True
    else:
        return False


def process_data_train(pkl_file_path, save_path, th_iou_train):
    all_data = pickle.load(open(pkl_file_path, "rb"))
    all_gt_ls = get_all_gt(all_data)

    train_sentences = []
    train_sentence_info = []
    train_roi = []
    train_cls = []
    train_tbbox = []
    train_norm_tbbox = []

    sample_num = len(all_data)
    for sample_index in range(sample_num):
        gt_boxes = all_data[sample_index]["ground_truth_bbox"]
        gt_boxes = np.array([list(item) for item in gt_boxes])
        gt_classes = all_data[sample_index]["ground_truth_cls"]
        bboxs = all_data[sample_index]["region_proposal"]
        # be matrix
        bboxs = np.array([list(item) for item in bboxs])
        nroi = len(bboxs)
        sentence_matrix = all_data[sample_index]["sentence"]
        sentence_str_ls = all_data[sample_index]["str"].split(" ")
        ious = calc_ious_1d(bboxs, gt_boxes)

        rbbox = bboxs

        max_ious = ious.max(axis=1)
        max_idx = ious.argmax(axis=1)
        tbbox = bbox_transform_1d(bboxs, gt_boxes[max_idx])
        # lb_tbbox = lb_bbox_transform_1d(bboxs, gt_boxes[max_idx])

        pos_idx = []
        neg_idx = []

        for roi_index in range(nroi):
            gid = len(train_roi)

            if max_ious[roi_index] >= th_iou_train:
                pos_idx.append(gid)
                train_cls.append(gt_classes[max_idx[roi_index]])
                train_norm_tbbox.append(gid)
                train_roi.append(rbbox[roi_index])
                train_tbbox.append(tbbox[roi_index])
            else:
                if check_global_nega(all_gt_ls, rbbox[roi_index], all_data[sample_index]["str"], th_iou_train):
                    nega_num[0] += 1
                    if gt_classes[max_idx[roi_index]] == 1:
                        cover_nega.write("原句：\n")
                        cover_nega.write(" ".join(sentence_str_ls) + "\n")
                        # cover_nega.write("Ground Truth：\n")
                        # cover_nega.write(
                        #     " ".join(
                        #         sentence_str_ls[
                        #         gt_boxes[max_idx[roi_index]][0]:gt_boxes[max_idx[roi_index]][1] + 1]) + "\n")
                        cover_nega.write("负样本：\n")
                        cover_nega.write(" ".join(sentence_str_ls[bboxs[roi_index][0]:bboxs[roi_index][1] + 1]) + "\n")
                        cover_nega.write("\n\n")

                    train_roi.append(rbbox[roi_index])
                    train_tbbox.append(tbbox[roi_index])
                    neg_idx.append(gid)
                    train_cls.append(0)

        pos_idx = np.array(pos_idx)
        neg_idx = np.array(neg_idx)
        train_sentences.append(sentence_matrix)
        train_sentence_info.append({
            "pos_idx": pos_idx,
            "neg_idx": neg_idx,
        }
        )

    train_sentences = np.array(train_sentences)
    train_sentence_info = np.array(train_sentence_info)
    train_roi = np.array(train_roi)
    train_cls = np.array(train_cls)
    train_tbbox = np.array(train_tbbox)
    # train_lb_tbbox = np.array(train_lb_tbbox)
    train_norm_tbbox = feature_scaling(train_tbbox, train_norm_tbbox, True)

    np.savez(open(save_path, 'wb'),
             train_sentences=train_sentences, train_sentence_info=train_sentence_info,
             train_roi=train_roi, train_cls=train_cls, train_tbbox=train_tbbox,
             train_norm_tbbox=train_norm_tbbox)
    print("save in ", save_path)


def process_data_train_k_fold():
    th_iou_train = 0.7
    train_pkl_folder = 'dataset/train/train_relabeled_data_pkl_11_16_rb_modify/'
    save_folder = 'dataset/train/global_nega_train_relabeled_data_npz_11_16_rb_modify/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    ls_pkl_file_path = [train_pkl_folder + pkl for pkl in os.listdir(train_pkl_folder)]
    for i in range(len(ls_pkl_file_path)):
        save_path = save_folder + "train_th_iou_" + str(th_iou_train) + str(i + 1) + ".npz"
        process_data_train(ls_pkl_file_path[i], save_path, th_iou_train)
    print("th_iou_train", th_iou_train)


def process_data_test(pkl_file_path, save_path):
    all_data = pickle.load(open(pkl_file_path, "rb"))
    test_sentences = []
    test_sentence_info = []
    test_roi = []
    sample_num = len(all_data)
    for sample_index in range(sample_num):
        gt_boxes = all_data[sample_index]["ground_truth_bbox"]
        gt_boxes = np.array([list(item) for item in gt_boxes])
        gt_classes = all_data[sample_index]["ground_truth_cls"]
        bboxs = all_data[sample_index]["region_proposal"]
        gt_str = all_data[sample_index]["str"].split(" ")
        # be matrix
        bboxs = np.array([list(item) for item in bboxs])
        nroi = len(bboxs)
        sentence_matrix = all_data[sample_index]["sentence"]
        roi_ids = []
        for roi_index in range(nroi):
            glo_id = len(test_roi)
            test_roi.append(bboxs[roi_index])
            roi_ids.append(glo_id)

        roi_ids = np.array(roi_ids)
        test_sentences.append(sentence_matrix)
        test_sentence_info.append({
            "roi_ids": roi_ids,
            "gt_bboxs": gt_boxes,
            "gt_str": gt_str,
            "gt_cls": gt_classes})
    test_sentences = np.array(test_sentences)
    test_sentence_info = np.array(test_sentence_info)
    test_roi = np.array(test_roi)

    np.savez(open(save_path, 'wb'),
             test_sentences=test_sentences, test_sentence_info=test_sentence_info, test_roi=test_roi)


def process_data_test_k_fold():
    train_pkl_folder = 'dataset/test/test_relabeled_data_pkl_11_16_rb_modify/'
    save_folder = 'dataset/test/test_relabeled_data_npz_11_16_rb_modify/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    ls_pkl_file_path = [train_pkl_folder + pkl for pkl in os.listdir(train_pkl_folder)]
    for i in range(len(ls_pkl_file_path)):
        save_path = save_folder + "test" + str(i + 1) + ".npz"
        process_data_test(ls_pkl_file_path[i], save_path)


def main():
    # process_data_test_k_fold()
    process_data_train_k_fold()


if __name__ == '__main__':
    main()
    print("all_mean", all_mean)
    print("all_deviation", all_deviation)
    print("all_nega", nega_num)
