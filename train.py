import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys

sys.path.append('/TA-SSM /mamba')
from TA-SSM import VisionMamba3D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataset1 import MRIDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
# ssh -p 21616 root@i-1.gpushare.com
# X7Fr6zafkbE5mbBsRQfwd6rqYBDFzY67

def train_model(train_loader, test_loader, num_epochs=300, learning_rate=0.001, save_path='model_weights',
                pretrained_weights=True):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

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
        bimamba_type="v3",
        if_cls_token=True,
        use_double_cls_token=False,
        use_middle_cls_token=True
    ).to(torch.float32)

    model.to(device)

    # 加载预训练权重
    if pretrained_weights:
        model.load_state_dict(torch.load("/hy-tmp/baosichen/mamba/new/model_weights/model_epoch_17__acc:0.525.pth"))
        print(f"Loaded pretrained weights from {pretrained_weights}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_predictions = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())

        train_acc, train_prec, train_rec, train_f1 = calculate_metrics(all_labels, all_predictions)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}, '
              f'Accuracy: {train_acc}, Precision: {train_prec}, Recall: {train_rec}, F1 Score: {train_f1}')



        # 在测试集上评估模型
        model.eval()
        test_labels = []
        test_predictions = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

                preds = torch.argmax(outputs, dim=1)
                test_labels.extend(labels.cpu().numpy())
                test_predictions.extend(preds.cpu().numpy())

        test_acc, test_prec, test_rec, test_f1 = calculate_metrics(test_labels, test_predictions)
        print(f'Test Accuracy: {test_acc}, Precision: {test_prec}, Recall: {test_rec}, F1 Score: {test_f1}')

        class_accuracy = calculate_class_metrics(test_labels, test_predictions)
        print(f'Class-wise accuracy: {class_accuracy}')
        # 保存模型权重
        torch.save(model.state_dict(), os.path.join(save_path, f'model_epoch_{epoch + 1}__acc:{test_acc}.pth'))

# 加载数据
train_dataset = MRIDataset('/hy-tmp/baosichen/data/AD400vsCN400.mat', '/hy-tmp/baosichen/data/ADvsCN_all/',
                           train=True)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)

test_dataset = MRIDataset('/hy-tmp/baosichen/data/AD400vsCN400.mat', '/hy-tmp/baosichen/data/ADvsCN_all/',
                          train=False)
test_loader = DataLoader(test_dataset, batch_size=42, shuffle=False)


train_model(train_loader, test_loader)
