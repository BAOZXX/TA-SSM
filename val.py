import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys

sys.path.append('/hy-tmp/baosichen/mamba')
from Vim3D import VisionMamba3D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataset_test import MRIDataset
from sklearn.metrics import confusion_matrix, roc_auc_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calculate_confusion_matrix_metrics(labels, predictions):
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    # 计算各项指标
    acc = (tp + tn) / (tp + tn + fp + fn)
    sen = tp / (tp + fn)  # 灵敏度
    spe = tn / (tn + fp)  # 特异性
    auc = roc_auc_score(labels, predictions)  # AUC

    return tp, tn, fp, fn, acc, sen, spe, auc
def calculate_metrics(labels, predictions):
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, average='weighted')
    rec = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    return acc, prec, rec, f1


def calculate_class_metrics(test_labels, test_predictions):
    class_correct = {0: 0, 1: 0}
    class_total = {0: 0, 1: 0}
    for label, prediction in zip(test_labels, test_predictions):
        class_total[label] += 1
        if label == prediction:
            class_correct[label] += 1

    class_accuracy = {cls: class_correct[cls] / class_total[cls] for cls in class_correct}
    return class_accuracy


def train_model( test_loader, learning_rate=0.001, save_path='model_weights',
                pretrained_weights = True):

    model = VisionMamba3D(
        img_size=256,
        patch_size=16,
        stride=16,
        depth=24,
        embed_dim=768,
        channels=1,
        num_classes=2,
        ssm_cfg=None,
        drop_rate=0.,
        drop_path_rate=0.1,
        norm_epsilon=1e-5,
        rms_norm=True,
        fused_add_norm=True,
        residual_in_fp32=True,
        pt_hw_seq_len=14,
        if_bidirectional=False,
        final_pool_type='mean',
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        flip_img_sequences_ratio=-1.,
        if_bimamba=True,
        bimamba_type="v2",
        if_cls_token=True,
        use_double_cls_token=False,
        use_middle_cls_token=True
    ).to(torch.float32)

    model.to(device)

    # 加载预训练权重
    if pretrained_weights:
        model.load_state_dict(torch.load("/hy-tmp/baosichen/mamba/new/model_weights/MCI/MCI-2-73.493.pth"))
        print(f"Loaded pretrained weights from {pretrained_weights}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)



    # 在测试集上评估模型
    model.eval()
    test_labels = []
    test_predictions = []
    incorrect_filenames = []  # 存储预测错误的文件名
    with torch.no_grad():
        for inputs, labels, filenames in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            preds = torch.argmax(outputs, dim=1)
            test_labels.extend(labels.cpu().numpy())
            test_predictions.extend(preds.cpu().numpy())
            # 记录预测错误的文件名
            for label, prediction, filename in zip(labels.cpu().numpy(), preds.cpu().numpy(), filenames):
                if label != prediction:
                    incorrect_filenames.append(filename)
    tp, tn, fp, fn, acc, sen, spe, auc = calculate_confusion_matrix_metrics(test_labels, test_predictions)
    print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
    print(f'Accuracy (ACC): {acc}, Sensitivity (SEN): {sen}, Specificity (SPE): {spe}, AUC: {auc}')
    test_acc, test_prec, test_rec, test_f1 = calculate_metrics(test_labels, test_predictions)
    print(f'Test Accuracy: {test_acc}, Precision: {test_prec}, Recall: {test_rec}, F1 Score: {test_f1}')

    class_accuracy = calculate_class_metrics(test_labels, test_predictions)
    print(f'Class-wise accuracy: {class_accuracy}')
    # 输出预测错误的文件名
    print(f'Incorrectly predicted files: {incorrect_filenames}')


# 加载数据


train_dataset = MRIDataset('/hy-tmp/baosichen/data/MCI400vsCN400.mat', '/hy-tmp/baosichen/data/MCvsCN_400vs400',
                          train=True)
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=False)
test_dataset = MRIDataset('/hy-tmp/baosichen/data/MCI400vsCN400.mat', '/hy-tmp/baosichen/data/MCvsCN_400vs400',
                          train=False)
test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)

train_model(train_loader)
train_model(test_loader)
