import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from functools import partial
from einops import rearrange
from SegMamba.mamba.mamba_ssm.modules.mamba_simple import Mamba
from Vim3D import VisionMamba3D
from new.dataset_test import MRIDataset

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


class ModifiedMamba(Mamba):
    def forward(self, hidden_states, inference_params=None):
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out, None, None, None

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())

        A_b = -torch.exp(self.A_b_log.float())
        out_a = mamba_inner_fn_no_out_proj(
            xz,
            self.conv1d.weight,
            self.conv1d.bias,
            self.x_proj.weight,
            self.dt_proj.weight,
            A,
            None,
            None,
            self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        out_b = mamba_inner_fn_no_out_proj(
            xz.flip([-1]),
            self.conv1d_b.weight,
            self.conv1d_b.bias,
            self.x_proj_b.weight,
            self.dt_proj_b.weight,
            A_b,
            None,
            None,
            self.D_b.float(),
            delta_bias=self.dt_proj_b.bias.float(),
            delta_softplus=True,
        )

        A_s = -torch.exp(self.A_s_log.float())
        cls_token = xz[:, :, 4096:]
        xz = xz[:, :, :4096]
        xz_s = xz.chunk(self.nslices, dim=-1)
        xz_s = torch.stack(xz_s, dim=-1)
        xz_s = xz_s.flatten(-2)
        xz_s = torch.cat((xz_s, cls_token), dim=-1)
        out_s = mamba_inner_fn_no_out_proj(
            xz_s,
            self.conv1d_s.weight,
            self.conv1d_s.bias,
            self.x_proj_s.weight,
            self.dt_proj_s.weight,
            A_s,
            None,
            None,
            self.D_s.float(),
            delta_bias=self.dt_proj_s.bias.float(),
            delta_softplus=True,
        )
        cls_token = out_s[:, :, 4096:]
        out_s = out_s[:, :, :4096]
        out_s = out_s.reshape(batch, self.d_inner, seqlen // self.nslices, self.nslices).permute(0, 1, 3, 2).flatten(-2)
        out_s = torch.cat((out_s, cls_token), dim=-1)

        out = F.linear(rearrange(out_a + out_b.flip([-1]) + out_s, "b d l -> b l d"), self.out_proj.weight,
                       self.out_proj.bias)

        return out, out_a, out_b, out_s


class FeatureExtractorHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.feature = None
        self.gradient = None

    def hook_fn(self, module, input, output):
        self.feature = output
        output.register_hook(self.save_gradient)

    def save_gradient(self, grad):
        self.gradient = grad

    def close(self):
        self.hook.remove()


def compute_gradcam(feature_extractor, class_idx=None):
    gradients = feature_extractor.gradient[0]
    activations = feature_extractor.feature[0]

    weights = torch.mean(gradients, dim=(1, 2))
    cam = torch.zeros(activations.shape[1:], dtype=activations.dtype)

    for i, w in enumerate(weights):
        cam += w * activations[i, :, :]

    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / cam.max()
    return cam


def visualize_gradcam(gradcam, input_image, title='Grad-CAM'):
    plt.imshow(to_pil_image(input_image.cpu()), alpha=0.5)
    plt.imshow(gradcam.cpu().detach().numpy(), cmap='jet', alpha=0.5)
    plt.colorbar()
    plt.title(title)
    plt.show()


    def forward(self, x):
        return self.mamba_layer(x)

def load_sample_data(img_size=256):
    return torch.randn(1, 1, img_size, img_size, img_size)

def load_model_weights(model, path):
    model.load_state_dict(torch.load(path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionMamba3D(
    device=device,
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
    bimamba_type="v3a",
    if_cls_token=True,
    use_double_cls_token=False,
    use_middle_cls_token=True
).to(torch.float32)
model.to(device)

# 假设你的模型权重保存在 'model_weights.pth'
weights_path = '/hy-tmp/baosichen/mamba/new/model_weights/MCI-3-att-80.pth'
load_model_weights(model, weights_path)

x = load_sample_data().to(device)
# Create feature extractor hooks
out_a_hook = FeatureExtractorHook(model.mamba_layer)
out_b_hook = FeatureExtractorHook(model.mamba_layer)
out_s_hook = FeatureExtractorHook(model.mamba_layer)

# Forward pass
model.eval()
with torch.no_grad():
    out, out_a, out_b, out_s = model(x)

# Compute Grad-CAM
gradcam_a = compute_gradcam(out_a_hook)
gradcam_b = compute_gradcam(out_b_hook)
gradcam_s = compute_gradcam(out_s_hook)

# Visualize Grad-CAM
visualize_gradcam(gradcam_a, x[0][0], title='Grad-CAM out_a')
visualize_gradcam(gradcam_b, x[0][0], title='Grad-CAM out_b')
visualize_gradcam(gradcam_s, x[0][0], title='Grad-CAM out_s')
def train_model(test_loader, learning_rate=0.001, save_path='model_weights', pretrained_weights=True):
    model = VisionMamba3D(
        device = device,
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
        bimamba_type="v3a",
        if_cls_token=True,
        use_double_cls_token=False,
        use_middle_cls_token=True
    ).to(torch.float32)

    model.to(device)

    if pretrained_weights:
        model.load_state_dict(torch.load("/hy-tmp/baosichen/mamba/new/model_weights/MCI-3-att-80.pth"))
        print(f"Loaded pretrained weights from {pretrained_weights}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.eval()
    test_labels = []
    test_predictions = []
    incorrect_filenames = []  # 存储预测错误的文件名
    with torch.no_grad():
        for inputs, labels, filenames in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Create feature extractor hooks
            out_a_hook = FeatureExtractorHook(model.mamba_layer)
            out_b_hook = FeatureExtractorHook(model.mamba_layer)
            out_s_hook = FeatureExtractorHook(model.mamba_layer)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            test_labels.extend(labels.cpu().numpy())
            test_predictions.extend(preds.cpu().numpy())

            # Compute Grad-CAM for one batch and visualize
            gradcam_a = compute_gradcam(out_a_hook)
            gradcam_b = compute_gradcam(out_b_hook)
            gradcam_s = compute_gradcam(out_s_hook)
            visualize_gradcam(gradcam_a, inputs[0][0], title='Grad-CAM out_a')
            visualize_gradcam(gradcam_b, inputs[0][0], title='Grad-CAM out_b')
            visualize_gradcam(gradcam_s, inputs[0][0], title='Grad-CAM out_s')

            # 记录预测错误的文件名
            for label, prediction, filename in zip(labels.cpu().numpy(), preds.cpu().numpy(), filenames):
                if label != prediction:
                    incorrect_filenames.append(filename)

    test_acc, test_prec, test_rec, test_f1 = calculate_metrics(test_labels, test_predictions)
    print(f'Test Accuracy: {test_acc}, Precision: {test_prec}, Recall: {test_rec}, F1 Score: {test_f1}')

    class_accuracy = calculate_class_metrics(test_labels, test_predictions)
    print(f'Class-wise accuracy: {class_accuracy}')
    print(f'Incorrectly predicted files: {incorrect_filenames}')

train_dataset = MRIDataset('/hy-tmp/baosichen/data/MCI400vsCN400.mat', '/hy-tmp/baosichen/data/MCvsCN_400vs400', train=True)
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=False)
test_dataset = MRIDataset('/hy-tmp/baosichen/data/MCI400vsCN400.mat', '/hy-tmp/baosichen/data/MCvsCN_400vs400', train=False)
test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)

train_model(test_loader)
