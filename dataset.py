import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk

class MRIDataset(Dataset):
    def __init__(self, mat_file, data_dir, train=True):
        # 加载.mat文件
        mat_data = sio.loadmat(mat_file)
        if train:
            self.samples = mat_data['samples_train'].flatten()
            self.labels = mat_data['labels_train'].flatten()
        else:
            self.samples = mat_data['samples_test'].flatten()
            self.labels = mat_data['labels_test'].flatten()
        self.data_dir = data_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]  # 获取样本路径
        img_path = os.path.join(self.data_dir, sample_path)  # 构建完整路径

        label = self.labels[idx]

        # 读取.nii.gz文件
        img = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(img).astype(np.float32)

        # 将数据转换为torch张量
        img_tensor = torch.tensor(img_array).unsqueeze(0)  # 添加通道维度

        return img_tensor, torch.tensor(label, dtype=torch.long)
