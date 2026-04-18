import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_classif
import torch
import torch.nn as nn

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class GBS(nn.Module):
    def __init__(self, data, patch_label, patch_data, num_bands, top_k, **kwargs):
        """
        初始化GBS类
        :param data: 高光谱图像数据 (n_samples, n_bands)
        :param label: 地面真实标签 (n_samples,)
        :param top_k: 基于贪心选择策略选择的前N个波段
        """
        super(GBS, self).__init__()
        self.data = data
        self.patch_label = patch_label
        self.patch_data = patch_data
        self.num_bands = num_bands
        self.top_k = top_k

        # Calculate PCC and MI
        self.pcc_matrix = self.calculate_pcc()
        self.mi = self.calculate_mi()

        # Select bands using greedy strategy
        self.selected_bands = self.greedy_band_selection(self.pcc_matrix, self.mi, self.top_k)

    # 计算所有波段间的Pearson相关系数
    def calculate_pcc(self):
        # _, _, num_bands = data.shape
        pcc_matrix = np.zeros((self.num_bands, self.num_bands))
        for i in range(self.num_bands):
            for j in range(i, self.num_bands):
                if i == j:
                    pcc_matrix[i, j] = 1.0  # 自身的相关性为1
                else:
                    corr, _ = pearsonr(self.data[:, :, i].flatten(), self.data[:, :, j].flatten())
                    pcc_matrix[i, j] = corr
                    pcc_matrix[j, i] = corr  # 对称矩阵
        return pcc_matrix

    # 计算每个波段与标签之间的互信息
    def calculate_mi(self):
        # 将patch_data展平成二维数组，每个样本对应一个cube的所有波段
        n_samples, height, width, n_bands = self.patch_data.shape
        data_reshaped = self.patch_data.reshape(n_samples, -1)
        mi = mutual_info_classif(data_reshaped, self.patch_label)
        # 将MI值平均到每个波段上
        mi_per_band = mi.reshape(height * width, n_bands).mean(axis=0)
        return mi_per_band

    def greedy_band_selection(self, pcc_matrix, mi, k):
        selected_bands = []
        available_bands = list(range(len(mi)))
        
        while len(selected_bands) < k and available_bands:
            best_gain = -np.inf
            best_band = None

            for band in available_bands:
                # 跳过索引0的波段
                if band == 0:
                    continue
                
                # 如果这是第一个波段，则直接选取MI最高的（除了索引0）
                if not selected_bands:
                    gain = mi[band]
                else:
                    # 计算新加入波段后的整体性能增益
                    avg_pcc = np.mean([pcc_matrix[band, b] for b in selected_bands])
                    gain = mi[band] - avg_pcc

                if gain > best_gain:
                    best_gain = gain
                    best_band = band

            if best_band is not None:
                selected_bands.append(best_band)
                available_bands.remove(best_band)

        return selected_bands
    
    def forward(self, x, **kwargs):
        selected_x = x[:, :, :, self.selected_bands]
        return selected_x