# -*- coding:utf-8 -*-
#
# Created by ETIP-team
#
# On 2018-09-04


import os
import re
import time
import tensorflow as tf
import xml.etree.ElementTree as ET
import model
import config as cfg


# train_path = "训练样本\\"

# train_path = "训练样本\\"   #  todo  改写成由dataset里面训练与检测的。
detect_path = dataset_path = "data/"
k_fold = cfg.K_FOLD

CNN = model.CNNTextClassifier()  # create the CNN tensor graph
sess = tf.InteractiveSession()  # create the default interactive session
nn_save_path = cfg.NN_SAVE_PATH
all_write_path = cfg.ALL_WRITE_PATH

train_flag = cfg.TRAIN_FLAG
if train_flag:
    print("Start Training")
    ls_detect_folder = os.listdir(detect_path)
    for fold_index in range(k_fold):
        saver = tf.train.Saver(max_to_keep=len(model.ls_cross_entropy_th)+2)
        time_start = time.time()
        # read training samples from xml files
        # BATCH TRAIN
        train_times = 0
        train_data_path = model.create_train_path(dataset_path, fold_index)
        ls_all_samples = model.create_train_samples_batch(train_data_path)
        ls_train = model.get_train_batch(ls_all_samples)
        ls_train_set = model.training_set(ls_train)
        tf.global_variables_initializer().run()
        for i in range(len(model.ls_cross_entropy_th)):
            time_start = time.time()
            flag_train = True
            while model.cnn_batch_train(CNN, ls_train_set, train_times, model.ls_cross_entropy_th[i]):
                ls_train = model.get_train_batch(ls_all_samples)
                ls_train_set = model.training_set(ls_train)
                train_times += 1
            str_entropy = str(model.ls_cross_entropy_th[i]).ljust(4, '0')  # 0.1 -> 0.10
            check_point_save_path = nn_save_path + str(fold_index+1) +"/cross_entropy"+str_entropy
            if not os.path.exists(check_point_save_path):
                os.makedirs(check_point_save_path)
            save_path = saver.save(sess, check_point_save_path+"/model.ckpt")
            print("Model saved in path: %s" % save_path)
            # detect
            one_fold_write_path = all_write_path+"cross_entropy"+str_entropy+"/result"+str(fold_index+1)+"/"
            if not os.path.exists(one_fold_write_path):
                os.makedirs(one_fold_write_path)
            for file in os.listdir(detect_path+ls_detect_folder[fold_index]+"/"):
                # todo write2function detect
                detect_file_path = detect_path + ls_detect_folder[fold_index]+"/"+file
                write_file_path = one_fold_write_path+file[:-4] +".txt"
                model.detect_file_and_write(detect_file_path, write_file_path, sess)
            print(one_fold_write_path + "    Write done!\n\n")
        time_end = time.time()
        time_elapsed = time_end - time_start
        print('The train and write time last {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        model.remove_train_path(train_data_path)

else:
    print("Start Detecting")
    saver = tf.train.Saver()
    # write_path = all_write_path[index_different_entropy]
    ls_detect_folder = os.listdir(detect_path)
    for fold_index in range(k_fold):
        # if index == 1:
        #     exit()
        for i in range(len(model.ls_cross_entropy_th)):
            str_entropy = str(model.ls_cross_entropy_th[i])
            if len(str_entropy) < len(str(0.12)):
                str_entropy += "0"
            load_path = nn_save_path + str(fold_index+1) +"/cross_entropy"+str_entropy+"/"
            print("Load Model from   " + load_path)
            saver = tf.train.import_meta_graph(load_path +"model.ckpt.meta")
            saver.restore(sess, tf.train.latest_checkpoint(load_path))
            tf.get_default_graph()
            print("Model is loaded")
            one_fold_write_path = all_write_path + "cross_entropy" + str_entropy + "/result" + str(fold_index + 1) + "/"
            if not os.path.exists(one_fold_write_path):
                os.makedirs(one_fold_write_path)
            for file in os.listdir(detect_path+ls_detect_folder[fold_index]+"/"):
                # print(str_entropy + "\\" + ls_write_folder[index] + "    Write done!\n\n")
                detect_file_path = detect_path + ls_detect_folder[fold_index] + "/" + file
                write_file_path = one_fold_write_path+file[:-4] +".txt"
                model.detect_file_and_write(detect_file_path, write_file_path, sess)
            print(one_fold_write_path + "    Write done!\n\n")
