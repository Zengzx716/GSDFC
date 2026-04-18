import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from .DWT_2D import DWT_2D
from .DWT_3D_N import DWT_3D
from .ema import GSAM


def seq2img(x):
    [b, c, d] = x.shape
    p = int(d ** .5)
    x = x.reshape((b, c, p, p))
    return x


# 空间-光谱注意力模块
class Spatial_Spectral_Attn_3d(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spatial_Spectral_Attn_3d, self).__init__()
        self.attn = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)
        y = self.attn(y)

        return x * y.expand_as(x)

# 空间-光谱特征提取分支
class HSI_Encoder_3D(nn.Module):
    def __init__(self, in_depth, patch_size, wavename,
                 in_channels_3d=1, out_channels_3d=16, out_channels_2d=64, attn_kernel_size=7):
        super(HSI_Encoder_3D, self).__init__()
        self.in_depth = in_depth
        self.patch_size = patch_size

        # DWT 3d
        self.DWT_layer_3D = DWT_3D(wavename=wavename)

        # 3d cnn for x_lll
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels_3d, out_channels=out_channels_3d // 2, kernel_size=(3, 3, 3),  # 减少到原来的1/4
                      stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels_3d // 2),
            nn.ReLU(),
        )

        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(in_channels=out_channels_3d // 2, out_channels=out_channels_3d, kernel_size=(3, 3, 3), # 减少到原来的1/2
                      stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels_3d),
            nn.ReLU(),
        )

        self.SS_attn = Spatial_Spectral_Attn_3d(kernel_size=attn_kernel_size)

        # 3d cnn for high components
        self.conv3d_high = nn.Sequential(
            nn.Conv3d(in_channels=in_channels_3d * 7, out_channels=out_channels_3d, kernel_size=1),
            nn.BatchNorm3d(out_channels_3d),
            nn.ReLU(),
        )

        # 2d cnn for all components
        self.in_channels_2d = int(self.get_inchannels_2d())
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels_2d, out_channels=out_channels_2d, kernel_size=1),
            nn.BatchNorm2d(out_channels_2d),
            nn.ReLU(),
        )

    def get_inchannels_2d(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.in_depth // 2, self.patch_size // 2, self.patch_size // 2))
            x = self.conv3d_1(x)
            x = self.conv3d_2(x)

            x = torch.cat([x, x], dim=1)
            _, t, c, _, _ = x.size()
        return t * c

    def forward(self, hsi_img):
        # DWT 3d
        hsi_img = hsi_img.unsqueeze(1)
        x_dwt = self.DWT_layer_3D(hsi_img.permute(0, 1, 3, 2, 4))
        # # 打印 x_dwt 的每个元素的形状
        # for i, x in enumerate(x_dwt):
        #     print(f"x_dwt[{i}] shape: {x.shape}")

        # 3d cnn
        x_lll = x_dwt[0].permute(0, 1, 3, 2, 4)
        x_lll = self.conv3d_1(x_lll)
        x_lll = self.conv3d_2(x_lll)
        x_lll = self.SS_attn(x_lll)

        # 假设目标尺寸是 (batch_size, channels, depth, height, width)
        target_size = (x_dwt[0].size(0), x_dwt[0].size(1), x_dwt[0].size(2), x_dwt[0].size(3), x_dwt[0].size(4))

        # 调整所有张量到目标尺寸
        adjusted_x_dwt = [F.interpolate(x, size=target_size[2:], mode='trilinear', align_corners=False) if x.size() != target_size else x for x in x_dwt]

        # high frequency components processing
        x_high = torch.cat([x.permute(0, 1, 3, 2, 4) for x in adjusted_x_dwt[1:8]], dim=1)
        x_high = self.conv3d_high(x_high)
        x = torch.cat([x_lll, x_high], dim=1)

        # 2d cnn
        x = rearrange(x, 'b c d h w ->b (c d) h w')
        x = self.conv2d(x)

        return x

class HSI_Encoder_2D(nn.Module):
    def __init__(self, wavename, in_channels, out_channels=64, GSAM_factor=8):
        super(HSI_Encoder_2D, self).__init__()
        self.DWT_layer_2D = DWT_2D(wavename=wavename)

        # 2d cnn for x_ll
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        # GSAM module for attention
        self.S_attn = GSAM(out_channels, factor=GSAM_factor)

        # high frequency components processing
        self.conv_high = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 3, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        # 2d cnn for all components
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, hsi_img):
        x_dwt = self.DWT_layer_2D(hsi_img)

        # x_ll -> ch_select
        x_ll = x_dwt[0]
        x_ll = self.conv1(x_ll)
        x_ll = self.conv2(x_ll)
        x_ll = self.S_attn(x_ll)
        print('after GSAM 2d:', x_ll.shape)

        # high frequency component processing
        x_high = torch.cat([x for x in x_dwt[1:4]], dim=1)
        x_high = self.conv_high(x_high)

        x = torch.cat([x_ll, x_high], dim=1)
        x = self.conv2d(x)
        
        return x

class RectNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RectNet, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.deconv(x)
        return x

class DFCFFM(nn.Module):
    def __init__(self, l1, patch_size, wavename, attn_kernel_size,
                 GSAM_factor, coefficient_hsi, fae_embed_dim):
        super().__init__()
        self.weight_hsi = torch.nn.Parameter(torch.Tensor([coefficient_hsi]))

        self.hsi_encoder_3d = HSI_Encoder_3D(in_depth=l1, patch_size=patch_size,
                                             wavename=wavename, out_channels_2d=fae_embed_dim,
                                             attn_kernel_size=attn_kernel_size)

        # self.hsi_encoder_2d = HSI_Encoder_2D(wavename=wavename, in_channels=l1, out_channels=fae_embed_dim,
        #                                      attn_kernel_size=attn_kernel_size)
        self.hsi_encoder_2d = HSI_Encoder_2D(wavename=wavename, in_channels=l1, out_channels=fae_embed_dim,
                                             GSAM_factor=GSAM_factor)  
        self.rec = RectNet(fae_embed_dim, l1)

    # 前向传播
    def forward(self, img_hsi):
        # hsi encoder
        x_hsi_3d = self.hsi_encoder_3d(img_hsi)
        print("after 3d conv", x_hsi_3d.shape)
        x_hsi_2d = self.hsi_encoder_2d(img_hsi)
        print("after 2d conv", x_hsi_2d.shape)
        x_hsi = x_hsi_3d + x_hsi_2d
        print("after fusion", x_hsi.shape)
        x_cnn = self.weight_hsi * x_hsi
        x = x_cnn.flatten(2)
        print("after flatten", x.shape)
        x = seq2img(x)
        x_rec = self.rec(x)
        print("after reconstruction", x_rec.shape)

        return x_rec
