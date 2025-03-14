import os
import numpy as np
import scipy.io as sio
from random import sample

def ran_datasplit(task, img_path):
    sample_name = []
    labels = []
    if task == 1:
        for img in os.listdir(img_path + '/AD'):
            sample_name.append('AD/' + img)
            labels.append(1)
        for img in os.listdir(img_path + '/CN'):
            sample_name.append('CN/' + img)
            labels.append(0)
        task_name = 'AD_classification'

    elif task == 2:
        for img in os.listdir(img_path + '/MC'):
            sample_name.append('MC/' + img)
            labels.append(1)
        for img in os.listdir(img_path + '/CN'):
            sample_name.append('CN/' + img)
            labels.append(0)
        task_name = 'MCI_classification'
    elif task == 3:
        for img in os.listdir(img_path + '/MCI'):
            sample_name.append('MCI/' + img)
            labels.append(2)
        for img in os.listdir(img_path + '/AD'):
            sample_name.append('AD/' + img)
            labels.append(1)
        for img in os.listdir(img_path + '/CN'):
            sample_name.append('CN/' + img)
            labels.append(0)
        task_name = 'AD_MCI_CN'
    sample_name = np.array(sample_name)
    labels = np.array(labels)
    permut = np.random.permutation(len(sample_name))
    np.take(sample_name, permut, out=sample_name)
    np.take(labels, permut, out=labels)
    if task == 1:
        pos_list = np.arange(len(labels))[labels == 0]
        neg_list = np.arange(len(labels))[labels == 1]
        pos_test = sample(list(pos_list), round(len(pos_list) / 5))
        neg_test = sample(list(neg_list), round(len(neg_list) / 5))
        test_list = pos_test + neg_test
        test_list = sorted(test_list)

    elif task == 2:
        pos_list = np.arange(len(labels))[labels == 0]
        neg_list = np.arange(len(labels))[labels == 1]
        pos_test = sample(list(pos_list), round(len(pos_list) / 5))
        neg_test = sample(list(neg_list), round(len(neg_list) / 5))
        test_list = pos_test + neg_test
        test_list = sorted(test_list)
    train_list = list(set(range(len(sample_name))).difference(set(test_list)))
    samples_train = sample_name[train_list]
    labels_train = labels[train_list]
    samples_test = sample_name[test_list]
    labels_test = labels[test_list]
    sio.savemat('/hy-tmp/baosichen/data/{}400000.mat'.format(task_name), {"samples_train": samples_train, "samples_test": samples_test, "labels_train": labels_train, "labels_test": labels_test})
task = 1
img_path = "/data"
ran_datasplit(task, img_path)

