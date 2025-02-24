import os
import numpy as np
import scipy.io as sio
from random import sample

def datasplit(task, train_img_path, test_img_path):
    train_sample_name = []
    train_labels = []
    test_sample_name = []
    test_labels = []
    if task == 1:
        for img in os.listdir(train_img_path + '/AD'):
            train_sample_name.append('AD/' + img)
            train_labels.append(1)
        for img in os.listdir(train_img_path + '/CN'):
            train_sample_name.append('CN/' + img)
            train_labels.append(0)
        for img in os.listdir(test_img_path + '/AD'):
            test_sample_name.append('AD/' + img)
            test_labels.append(1)
        for img in os.listdir(test_img_path + '/CN'):
            test_sample_name.append('CN/' + img)
            test_labels.append(0)
        task_name = 'AD_classification'

    elif task == 2:
        for img in os.listdir(train_img_path + '/MC'):
            train_sample_name.append('MC/' + img)
            train_labels.append(1)
        for img in os.listdir(train_img_path + '/CN'):
            train_sample_name.append('CN/' + img)
            train_labels.append(0)
        for img in os.listdir(test_img_path + '/MC'):
            test_sample_name.append('MC/' + img)
            test_labels.append(1)
        for img in os.listdir(test_img_path + '/CN'):
            test_sample_name.append('CN/' + img)
            test_labels.append(0)
        task_name = 'MCI_classification'
    elif task == 3:
        for img in os.listdir(train_img_path + '/MCI'):
            train_sample_name.append('MCI/' + img)
            train_labels.append(2)
        for img in os.listdir(train_img_path + '/AD'):
            train_sample_name.append('AD/' + img)
            train_labels.append(1)
        for img in os.listdir(train_img_path + '/CN'):
            train_sample_name.append('CN/' + img)
            train_labels.append(0)
        for img in os.listdir(test_img_path + '/MCI'):
            test_sample_name.append('MCI/' + img)
            test_labels.append(2)
        for img in os.listdir(test_img_path + '/AD'):
            test_sample_name.append('AD/' + img)
            test_labels.append(1)
        for img in os.listdir(test_img_path + '/CN'):
            test_sample_name.append('CN/' + img)
            test_labels.append(0)
        task_name = 'AD_MCI_CN'
    # 将样本名称列表和标签列表转换为numpy数组
    train_sample_name = np.array(train_sample_name)
    train_labels = np.array(train_labels)
    test_sample_name = np.array(test_sample_name)
    test_labels = np.array(test_labels)
    if task == 1:
        pos_test = np.arange(len(test_labels))[test_labels == 0]
        neg_test = np.arange(len(test_labels))[test_labels == 1]
        test_list = pos_test + neg_test
        test_list = sorted(test_list)

        pos_train = np.arange(len(train_labels))[train_labels == 0]
        neg_train = np.arange(len(train_labels))[train_labels == 1]
        train_list = pos_train + neg_train
        train_list = sorted(train_list)

    elif task == 2:
        pos_test = np.arange(len(test_labels))[test_labels == 0]
        neg_test = np.arange(len(test_labels))[test_labels == 1]
        test_list = np.concatenate((pos_test, neg_test))

        pos_train = np.arange(len(train_labels))[train_labels == 0]
        neg_train = np.arange(len(train_labels))[train_labels == 1]
        train_list = np.concatenate((pos_train, neg_train))
    elif task == 3:
        neg2_test = np.arange(len(test_labels))[test_labels == 2]
        neg1_test = np.arange(len(test_labels))[test_labels == 1]
        pos_test = np.arange(len(test_labels))[test_labels == 0]
        test_list = pos_test + neg1_test + neg2_test
        test_list = sorted(test_list)

        neg2_train = np.arange(len(train_labels))[train_labels == 2]
        neg1_train = np.arange(len(train_labels))[train_labels == 1]
        pos_train = np.arange(len(train_labels))[train_labels == 0]
        train_list = pos_train + neg1_train + neg2_train
        train_list = sorted(train_list)


    # 根据索引获取训练集样本和标签
    samples_train = train_sample_name[train_list]
    labels_train = train_labels[train_list]
    # 根据索引获取测试集样本和标签
    samples_test = test_sample_name[test_list]
    labels_test = test_labels[test_list]
    # 将训练集和测试集的数据保存为.mat文件
    sio.savemat('/hy-tmp/baosichen/data/{}.mat'.format(task_name), {"samples_train": samples_train,
                                                                    "samples_test": samples_test,
                                                                    "labels_train": labels_train,
                                                                    "labels_test": labels_test})
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
    # 随机打乱样本的顺序
    permut = np.random.permutation(len(sample_name))
    # 按照随机顺序重新排列样本名称和标签
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
    elif task == 3:
        neg2_list = np.arange(len(labels))[labels == 2]
        neg1_list = np.arange(len(labels))[labels == 1]
        pos_list = np.arange(len(labels))[labels == 0]
        pos_test = sample(list(pos_list), round(len(pos_list) / 5))
        neg1_test = sample(list(neg1_list), round(len(neg1_list) / 5))
        neg2_test = sample(list(neg2_list), round(len(neg2_list) / 5))
        test_list = pos_test + neg1_test + neg2_test
        test_list = sorted(test_list)
    train_list = list(set(range(len(sample_name))).difference(set(test_list)))
    # 根据索引获取训练集样本和标签
    samples_train = sample_name[train_list]
    labels_train = labels[train_list]
    # 根据索引获取测试集样本和标签
    samples_test = sample_name[test_list]
    labels_test = labels[test_list]
    # 将训练集和测试集的数据保存为.mat文件
    sio.savemat('/hy-tmp/baosichen/data/{}400000.mat'.format(task_name), {"samples_train": samples_train,
                                                                    "samples_test": samples_test,
                                                                    "labels_train": labels_train,
                                                                    "labels_test": labels_test})
task = 1
# img_path = "/hy-tmp/baosichen/data/ADvsCN_300vs300"
# ran_datasplit(task, img_path)

train_img_path = "/hy-tmp/baosichen/data/ADvsCN/train"
test_img_path = "/hy-tmp/baosichen/data/ADvsCN/test"

datasplit(task, train_img_path, test_img_path)

