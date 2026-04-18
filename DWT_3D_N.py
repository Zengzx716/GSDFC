import torch
import numpy as np
import math
from torch.nn import Module
import pywt
from torch.autograd import Function

device = "cuda" if torch.cuda.is_available() else "cpu"


class DWT_3D(Module):
    """
    input: the 3D data to be decomposed -- (N, C, D, H, W)
    output: lfc -- (N, C, D/2, H/2, W/2)
            hfc_llh -- (N, C, D/2, H/2, W/2)
            hfc_lhl -- (N, C, D/2, H/2, W/2)
            hfc_lhh -- (N, C, D/2, H/2, W/2)
            hfc_hll -- (N, C, D/2, H/2, W/2)
            hfc_hlh -- (N, C, D/2, H/2, W/2)
            hfc_hhl -- (N, C, D/2, H/2, W/2)
            hfc_hhh -- (N, C, D/2, H/2, W/2)
    """

    def __init__(self, wavename):
        super(DWT_3D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self, input_depth, input_height, input_width):
        L1 = np.max((input_height, input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)

        # 生成低通滤波器矩阵
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                if index + j < matrix_h.shape[1]:  # 确保索引不越界
                    matrix_h[i, index + j] = self.band_low[j]
            index += 2

        # 生成高通滤波器矩阵
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                if index + j < matrix_g.shape[1]:  # 确保索引不越界
                    matrix_g[i, index + j] = self.band_high[j]
            index += 2

        # 裁剪矩阵以适应输入
        matrix_h_0 = matrix_h[0:(math.floor(input_height / 2)), 0:(input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(input_width / 2)), 0:(input_width + self.band_length - 2)]
        matrix_h_2 = matrix_h[0:(math.floor(input_depth / 2)), 0:(input_depth + self.band_length - 2)]

        matrix_g_0 = matrix_g[0:(input_height - math.floor(input_height / 2)),
                            0:(input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(input_width - math.floor(input_width / 2)),
                            0:(input_width + self.band_length - 2)]
        matrix_g_2 = matrix_g[0:(input_depth - math.floor(input_depth / 2)),
                            0:(input_depth + self.band_length - 2)]

        # 确保裁剪后的矩阵不会越界
        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_h_2 = matrix_h_2[:, (self.band_length_half - 1):end]

        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        matrix_g_2 = matrix_g_2[:, (self.band_length_half - 1):end]

        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).to(device)
            self.matrix_low_1 = torch.Tensor(matrix_h_1).to(device)
            self.matrix_low_2 = torch.Tensor(matrix_h_2).to(device)
            self.matrix_high_0 = torch.Tensor(matrix_g_0).to(device)
            self.matrix_high_1 = torch.Tensor(matrix_g_1).to(device)
            self.matrix_high_2 = torch.Tensor(matrix_g_2).to(device)
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_low_2 = torch.Tensor(matrix_h_2)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)
            self.matrix_high_2 = torch.Tensor(matrix_g_2)

    def forward(self, input):
        assert len(input.size()) == 5
        self.input_depth = input.size()[-3]
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix(self.input_depth, self.input_height, self.input_width)

        output = DWTFunction_3D.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_low_2,
                                      self.matrix_high_0, self.matrix_high_1, self.matrix_high_2)
        # 打印输出张量的形状
        for o in output:
            print(f"DWT output shape: {o.shape}")
        
        return output

class DWTFunction_3D(Function):
    @staticmethod
    def forward(ctx, input,
                matrix_Low_0, matrix_Low_1, matrix_Low_2,
                matrix_High_0, matrix_High_1, matrix_High_2):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_Low_2,
                              matrix_High_0, matrix_High_1, matrix_High_2)
        L = torch.matmul(matrix_Low_0, input)
        H = torch.matmul(matrix_High_0, input)
        LL = torch.matmul(L, matrix_Low_1).transpose(dim0=2, dim1=3)
        LH = torch.matmul(L, matrix_High_1).transpose(dim0=2, dim1=3)
        HL = torch.matmul(H, matrix_Low_1).transpose(dim0=2, dim1=3)
        HH = torch.matmul(H, matrix_High_1).transpose(dim0=2, dim1=3)
        LLL = torch.matmul(matrix_Low_2, LL).transpose(dim0=2, dim1=3)
        LLH = torch.matmul(matrix_Low_2, LH).transpose(dim0=2, dim1=3)
        LHL = torch.matmul(matrix_Low_2, HL).transpose(dim0=2, dim1=3)
        LHH = torch.matmul(matrix_Low_2, HH).transpose(dim0=2, dim1=3)
        HLL = torch.matmul(matrix_High_2, LL).transpose(dim0=2, dim1=3)
        HLH = torch.matmul(matrix_High_2, LH).transpose(dim0=2, dim1=3)
        HHL = torch.matmul(matrix_High_2, HL).transpose(dim0=2, dim1=3)
        HHH = torch.matmul(matrix_High_2, HH).transpose(dim0=2, dim1=3)
        return LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH

    @staticmethod
    def backward(ctx, grad_LLL, grad_LLH, grad_LHL, grad_LHH,
                 grad_HLL, grad_HLH, grad_HHL, grad_HHH):
        matrix_Low_0, matrix_Low_1, matrix_Low_2, matrix_High_0, matrix_High_1, matrix_High_2 = ctx.saved_variables
        grad_LL = torch.add(torch.matmul(matrix_Low_2.t(), grad_LLL.transpose(dim0=2, dim1=3)),
                            torch.matmul(matrix_High_2.t(), grad_HLL.transpose(dim0=2, dim1=3))).transpose(dim0=2,
                                                                                                           dim1=3)
        grad_LH = torch.add(torch.matmul(matrix_Low_2.t(), grad_LLH.transpose(dim0=2, dim1=3)),
                            torch.matmul(matrix_High_2.t(), grad_HLH.transpose(dim0=2, dim1=3))).transpose(dim0=2,
                                                                                                           dim1=3)
        grad_HL = torch.add(torch.matmul(matrix_Low_2.t(), grad_LHL.transpose(dim0=2, dim1=3)),
                            torch.matmul(matrix_High_2.t(), grad_HHL.transpose(dim0=2, dim1=3))).transpose(dim0=2,
                                                                                                           dim1=3)
        grad_HH = torch.add(torch.matmul(matrix_Low_2.t(), grad_LHH.transpose(dim0=2, dim1=3)),
                            torch.matmul(matrix_High_2.t(), grad_HHH.transpose(dim0=2, dim1=3))).transpose(dim0=2,
                                                                                                           dim1=3)
        grad_L = torch.add(torch.matmul(grad_LL, matrix_Low_1.t()), torch.matmul(grad_LH, matrix_High_1.t()))
        grad_H = torch.add(torch.matmul(grad_HL, matrix_Low_1.t()), torch.matmul(grad_HH, matrix_High_1.t()))
        grad_input = torch.add(torch.matmul(matrix_Low_0.t(), grad_L), torch.matmul(matrix_High_0.t(), grad_H))
        return grad_input, None, None, None, None, None, None, None, None
