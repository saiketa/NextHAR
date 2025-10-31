import random

import numpy as np
import torch
from openpyxl import Workbook
from openpyxl.styles import Font
import os
import itertools
from scipy.stats import special_ortho_group


def merge_dataset(data, label):
    index = np.zeros(data.shape[0], dtype=bool)
    label_new = []
    for i in range(label.shape[0]):
        temp_label = np.unique(label[i])
        if temp_label.size == 1:
            index[i] = True
            label_new.append(label[i, 0])
    return data[index], np.array(label_new)


def select_data(data, labels):
    user_label_index, position_label_index = 1, 2
    data = data[labels[:, 0, position_label_index] == 2, ...]
    labels = labels[labels[:, 0, position_label_index] == 2, ...]
    return data, labels


def split_data_label(data, labels, train_rate, valid_rate):
    # data, labels = select_data(data, labels)

    label_index = 0
    arr = np.arange(data.shape[0])
    np.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]
    train_num = int(data.shape[0] * train_rate)
    vali_num = int(data.shape[0] * valid_rate)
    data_train = data[:train_num, ...]
    data_vali = data[train_num:train_num + vali_num, ...]
    data_test = data[train_num + vali_num:, ...]
    t = np.min(labels[:, :, label_index])
    label_train = labels[:train_num, ..., label_index] - t
    label_vali = labels[train_num:train_num + vali_num, ..., label_index] - t
    label_test = labels[train_num + vali_num:, ..., label_index] - t

    data_train, label_train = merge_dataset(data_train, label_train)
    data_test, label_test = merge_dataset(data_test, label_test)
    data_vali, label_vali = merge_dataset(data_vali, label_vali)
    print('Train Size: %d, Vali Size: %d, Test Size: %d' % (
    label_train.shape[0], label_vali.shape[0], label_test.shape[0]))
    return data_train, label_train, data_vali, label_vali, data_test, label_test


def change_windowsize(data, label, window_size):
    # data (num, windowsize, feature) label (num, window_size, label_num)

    if data.shape[1] == window_size:
        return data, label

    print(f'change window_size from {data.shape[1]} to {window_size}')
    label_num = label.shape[2]
    label_unique_list = []
    for i in range(label_num):
        unique_value, count = np.unique(label[:, 0, i], return_counts=True)
        label_unique_list.append(unique_value.tolist())

    res_data = np.empty(shape=(0, window_size, data.shape[2]))
    res_label = np.empty(shape=(0, window_size, label_num))
    label_combines = list(itertools.product(*label_unique_list))
    for label_combine in label_combines:
        label_combine = np.array(label_combine)
        data_temp = data[(label[:, 0, :] == label_combine).all(axis=1)]

        data_temp = data_temp.reshape(-1, data.shape[2])
        redundancy = data_temp.shape[0] % window_size
        data_temp = data_temp[:-redundancy, :]
        data_temp = data_temp.reshape(-1, window_size, data.shape[2])

        label_temp = np.zeros(shape=(data_temp.shape[0], data_temp.shape[1], label_num))
        for i in range(len(label_combine)):
            label_temp[:, :, i] = label_combine[i]
        res_data = np.concatenate([res_data, data_temp], axis=0)
        res_label = np.concatenate([res_label, label_temp], axis=0)

    print(f'data shape after change window_size{res_data.shape}')
    print(f'label shape after change window_size{res_label.shape}')

    return res_data, res_label
