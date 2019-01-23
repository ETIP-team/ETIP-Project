# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2018-10-08


import pickle
import numpy as np
from utils import bbox_transform_1d, calc_ious_1d
import config as cfg

all_mean = []
all_deviation = []


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


def process_data_train(pkl_file_path, save_path, th_iou_train):
    all_data = pickle.load(open(pkl_file_path, "rb"))
    train_sentences = []
    train_sentence_info = []
    train_roi = []
    train_cls = []
    train_tbbox = []
    train_norm_tbbox = []

    # train_lb_tbbox = []
    sample_num = len(all_data)
    for sample_index in range(sample_num):
        gt_boxes = all_data[sample_index]["ground_truth_bbox"]
        gt_boxes = np.array([list(item) for item in gt_boxes])
        gt_classes = all_data[sample_index]["ground_truth_cls"]
        boxes = all_data[sample_index]["region_proposal"]
        # be matrix
        boxes = np.array([list(item) for item in boxes])
        roi_num = len(boxes)
        sentence_matrix = all_data[sample_index]["sentence"]
        # sentence_str_ls = all_data[sample_index]["str"].split(" ")
        # intersection, union = calc_intersection_union(boxes, gt_boxes)
        # ious = intersection / union
        # length_union_minus_intersection = union - intersection
        ious = calc_ious_1d(boxes, gt_boxes)

        real_bbox = boxes.copy()

        max_ious = ious.max(axis=1)
        max_idx = ious.argmax(axis=1)
        target_bbox = bbox_transform_1d(boxes, gt_boxes[max_idx])
        # lb_tbbox = lb_bbox_transform_1d(bboxs, gt_boxes[max_idx])

        pos_idx = []
        neg_idx = []

        for roi_index in range(roi_num):
            if max_ious[roi_index] < 0.1:
                continue
            # train_lb_tbbox.append(lb_tbbox[roi_index])
            # if gt_classes[max_idx[roi_index]] == 1:
            if max_ious[roi_index] >= th_iou_train:

                gid = len(train_roi)
                train_roi.append(real_bbox[roi_index])
                train_tbbox.append(target_bbox[roi_index])
                pos_idx.append(gid)
                train_cls.append(gt_classes[max_idx[roi_index]])
                train_norm_tbbox.append(gid)
            else:
                # cover_nega.write()
                # if 1 in gt_classes:
                #     all_gt_c_bbox = gt_boxes[np.where(np.array(gt_classes) == 1)]
                #     max_c_iou = calc_ious_1d(bboxs[roi_index, np.newaxis],
                #                              gt_boxes[np.where(np.array(gt_classes) == 1)]).max()
                #     if max_c_iou > 0:
                #         continue
                # if gt_classes[max_idx[roi_index]] == 1:
                # if True:
                #     cover_nega.write("原句：\n")
                #     cover_nega.write(" ".join(sentence_str_ls) + "\n")
                #     cover_nega.write("Ground Truth：\n")
                #     cover_nega.write(
                #         " ".join(
                #             sentence_str_ls[
                #             gt_boxes[max_idx[roi_index]][0]:gt_boxes[max_idx[roi_index]][1] + 1]) + "\n")
                #     cover_nega.write("负样本：\n")
                #     cover_nega.write(" ".join(sentence_str_ls[bboxs[roi_index][0]:bboxs[roi_index][1] + 1]) + "\n")
                #     cover_nega.write("\n\n")
                # cover_nega.write("")
                gid = len(train_roi)

                train_roi.append(real_bbox[roi_index])
                train_tbbox.append(target_bbox[roi_index])
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
    train_norm_tbbox = feature_scaling(train_tbbox, train_norm_tbbox, True)

    np.savez(open(save_path, 'wb'),
             train_sentences=train_sentences, train_sentence_info=train_sentence_info,
             train_roi=train_roi, train_cls=train_cls, train_tbbox=train_tbbox,
             train_norm_tbbox=train_norm_tbbox)
    print("save in ", save_path)


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
        boxes = all_data[sample_index]["region_proposal"]
        gt_str_ls = all_data[sample_index]["str"].split(" ")
        # be matrix
        boxes = np.array([list(item) for item in boxes])
        roi_num = len(boxes)
        sentence_matrix = all_data[sample_index]["sentence"]
        roi_ids = []
        for roi_index in range(roi_num):
            global_id = len(test_roi)
            test_roi.append(boxes[roi_index])
            roi_ids.append(global_id)

        roi_ids = np.array(roi_ids)
        test_sentences.append(sentence_matrix)
        test_sentence_info.append({
            "roi_ids": roi_ids,
            "gt_bboxs": gt_boxes,
            "gt_str": gt_str_ls,
            "gt_cls": gt_classes})
    test_sentences = np.array(test_sentences)
    test_sentence_info = np.array(test_sentence_info)
    test_roi = np.array(test_roi)

    np.savez(open(save_path, 'wb'),
             test_sentences=test_sentences, test_sentence_info=test_sentence_info, test_roi=test_roi)


def main():
    # prepare train
    th_iou_train = 0.7
    # train_pkl_file_path = 'dataset/debug/debug.pkl'
    # save_folder = 'dataset/debug/train_debug_'
    train_pkl_file_path = 'dataset/train/train_base_max_cls_len{}.pkl'.format(cfg.MAX_CLS_LEN)
    save_folder = 'dataset/train/train_base_max_cls_len{}_'.format(cfg.MAX_CLS_LEN)

    save_path = save_folder + "train_th_iou_" + str(th_iou_train) + ".npz"
    print("th_iou_train", th_iou_train)
    process_data_train(train_pkl_file_path, save_path, th_iou_train)

    # prepare test
    test_pkl_file_path = f'dataset/test/test_base_max_cls_len{cfg.MAX_CLS_LEN}.pkl'
    save_folder = f'dataset/test/test_base_max_cls_len{cfg.MAX_CLS_LEN}'
    save_path = save_folder + ".npz"
    process_data_test(test_pkl_file_path, save_path)


if __name__ == '__main__':
    main()
    print("train_samples_all_mean", all_mean)
    print("train_samples_all_deviation", all_deviation)
