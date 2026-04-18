import torch
from torch import nn
import torch.nn.functional as F
#GitHub地址：https://github.com/YOLOonMe/GSAM-attention-module
#论文地址：https://arxiv.org/abs/2305.13563v2

class GSAM(nn.Module):
    def __init__(self, channels, factor=8):
        super(GSAM, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

# # 输入 B C H W,  输出 B C H W
# if __name__ == '__main__':
#     block = GSAM(64).cuda()
#     input = torch.rand(1, 64, 64, 64).cuda()
#     output = block(input)
#     print(input.size(), output.size())

class MHSA3D(nn.Module):
    def __init__(self, channels, num_heads=1):
        super(MHSA3D, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=(1, 1, 1), bias=False)
        self.qkv_conv = nn.Conv3d(channels * 3, channels * 3, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=channels * 3, bias=False)# 331
        self.project_out = nn.Conv3d(channels, channels, kernel_size=(1, 1, 1), bias=False)

    def forward(self, x):
        b, t, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1,  h * w * t)
        k = k.reshape(b, self.num_heads, -1,  h * w * t)
        v = v.reshape(b, self.num_heads, -1,  h * w * t)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, t, -1, h, w))
        return out