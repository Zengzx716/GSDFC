import argparse
import torch
import os
import time  # 添加时间计算模块
from thop import profile  # 添加FLOPs计算库
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import seaborn as sns
from BS.GPMBS import GPMBS
from BS.data import loadata, minmax_scale, createImageCubes, trPixel2Patch
from FDANet.DFCFFM import DFCFFM
# from generate_cls_map import generate_png, generate_iter, load_dataset, sampling

device = "cuda" if torch.cuda.is_available() else "cpu"

print("device", device)

# Parameter Setting
def parse_args():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--seed", type=int, default=100,
                        help="Random seed")
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epoch')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--flag', choices=['train', 'test'], default='test',
                        help='testing mark')
    parser.add_argument('--patch_size', type=int, default=8,
                        help='cnn input size')
    parser.add_argument('--wavename', type=str, default='db2',
                        help='type of wavelet')
    parser.add_argument('--attn_kernel_size', type=int, default=9,
                        help='')
    parser.add_argument('--GSAM_factor', type=int, default=8,
                        help='')
    parser.add_argument('--coefficient_hsi', type=float, default=0.8,
                        help='weight of HSI data in feature fusion')
    parser.add_argument('--fae_embed_dim', type=int, default=16,
                        help='number of channels in fae input data')
    parser.add_argument('--fae_depth', type=int, default=1,
                        help='depth of fae')
    args = parser.parse_args()
    return args

# 实现 extract_patches 函数（PyTorch 版本）
def extract_patches(image, patch_size):
    """
    从图像中提取图像块
    image: 形状为 (H, W, C) 的张量
    patch_size: 图像块大小
    返回: 形状为 (N, C, patch_size, patch_size) 的张量
    """
    # 调整维度顺序为 (C, H, W)
    image = image.permute(2, 0, 1).unsqueeze(0).contiguous()
    
    # 使用 unfold 提取图像块
    patches = image.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    
    # 调整维度顺序
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    patches = patches.reshape(-1, image.size(1), patch_size, patch_size)
    
    return patches

# 主函数
def main():
    total_start_time = time.time()
    args = parse_args()
    dataset_name = 'IP'
    top_k = 25
    num_iterations = 10
    group_size = 8
    patch_size = args.patch_size    
    # 加载数据并转换为 GPU 张量
    data, label = loadata(dataset_name)
    # 转换数据类型为 float32
    data = data.astype(np.float32)
    label = label.astype(np.int64)
    n_row, n_column, n_band = data.shape
    
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    label_tensor = torch.tensor(label, dtype=torch.long).to(device)
    
    # 使用 PyTorch 进行归一化
    flat_data = data_tensor.reshape(-1, n_band)
    min_vals = flat_data.min(dim=0)[0]
    max_vals = flat_data.max(dim=0)[0]
    norm_data = (flat_data - min_vals) / (max_vals - min_vals + 1e-8)
    norm_data = norm_data.reshape(n_row, n_column, n_band)
    
    # 创建图像块
    pad_width = patch_size // 2
    padded_data = F.pad(norm_data.permute(2, 0, 1), 
                       (pad_width, pad_width, pad_width, pad_width), 
                       mode='reflect').permute(1, 2, 0).contiguous()
    
    # 准备输入数据
    data_patch = extract_patches(padded_data, patch_size).to(device)  # 需要实现 extract_patches 函数
    # data_patch = trPixel2Patch(padded_data, patch_size).to(device)
    print(f"提取完成，图像块数量: {data_patch.size(0)}")
    # data_patch = data_patch.float().to(device)
    
    # 初始化模型
    dffnet = DFCFFM(l1=n_band, patch_size=args.patch_size, wavename=args.wavename,
                    attn_kernel_size=args.attn_kernel_size, GSAM_factor=args.GSAM_factor,
                    coefficient_hsi=args.coefficient_hsi, fae_embed_dim=args.fae_embed_dim).to(device)
    
    # 计算复杂度
    total_params = sum(p.numel() for p in dffnet.parameters())
    print(f"模型参数量: {total_params/1000:.2f}K")
    
    # with torch.no_grad():
    #     flops, _ = profile(dffnet, inputs=(data_patch[:1],), verbose=False)
    # print(f"模型FLOPs: {flops:,}")
    # 使用小批量计算FLOPs
    with torch.no_grad():
        sample_batch = data_patch[:1]
        flops, _ = profile(dffnet, inputs=(sample_batch,), verbose=False)
        flops_per_patch = flops
        total_flops = flops_per_patch * data_patch.size(0)
    print(f"每块FLOPs: {flops_per_patch/1e6:.2f}M, 总FLOPs: {total_flops/1e6:.2f}M")
    
    # 特征融合
    fusion_start = time.time()
    dffnet.eval()  # 设置为评估模式，减少内存使用

    # 动态批大小计算
    free_mem = torch.cuda.mem_get_info()[0] / (1024**3)  # 空闲内存(GB)
    print(f"可用GPU内存: {free_mem:.2f}GB")
    
    # 估计每批所需内存 - 根据模型大小调整
    required_per_batch = 0.1  # 保守估计每批0.1GB
    batch_size = max(1, int(free_mem * 0.7 / required_per_batch))
    print(f"自动计算批大小: {batch_size}")
    fused_data_list = []

    with torch.no_grad():
        for i in range(0, data_patch.size(0), batch_size):
            start_idx = i
            end_idx = min(i + batch_size, data_patch.size(0))
            batch = data_patch[start_idx:end_idx]
            
            fused_batch = dffnet(batch)
            fused_data_list.append(fused_batch.cpu())  # 移到CPU内存保存
            
            # 清除中间变量释放内存
            del batch, fused_batch
            torch.cuda.empty_cache()
            
            # 进度显示
            progress = (end_idx / data_patch.size(0)) * 100
            print(f"融合进度: {progress:.1f}% ({end_idx}/{data_patch.size(0)})")

    # 在CPU上合并结果
    fused_data = torch.cat(fused_data_list, dim=0).to(device)
    # fused_data = dffnet(data_patch)
    fusion_time = time.time() - fusion_start
    print(f"特征融合时间: {fusion_time:.4f}秒")
    print('融合后数据形状:', fused_data.shape)

    # 波段选择
    band_selection_total_time = 0
    selected_bands = None

    # 减少迭代次数以节省内存
    reduced_iterations = min(num_iterations, 10)  # 根据内存情况调整

    for i in range(reduced_iterations):
        print(f"开始迭代 {i+1}/{reduced_iterations}")
        iter_start = time.time()
        
        # 减少数据传输
        gpmbs_data = fused_data.permute(0, 2, 3, 1)
        # 打印形状信息以便调试
        print(f"融合数据形状: {gpmbs_data.shape}")
        print(f"标签形状: {label_tensor.shape}")

        # 使用支持GPU的GPMBS
        model = GPMBS(data=data_tensor, 
                     patch_label=label_tensor, 
                     patch_data=gpmbs_data,
                     num_bands=n_band, 
                     top_k=top_k,
                     device=device)
        
        # 检查样本数量
        num_patches = gpmbs_data.size(0)
        num_labels = label_tensor.numel()  # 标签元素总数
        if num_patches != num_labels:
            print(f"警告: 图像块数量({num_patches})和标签数量({num_labels})不匹配!")
        # 可能需要进一步调整
        else:
            print("图像块数量和标签数量匹配")
        
        model.calculate_band_importance()
        selected_bands = model.selected_bands
        print("选择的波段:", selected_bands)
        
        iter_time = time.time() - iter_start
        band_selection_total_time += iter_time
        print(f"本次迭代时间: {iter_time:.4f}秒")
        
        # 清除中间变量
        del model
        torch.cuda.empty_cache()

    # 总时间
    total_time = time.time() - total_start_time
    print("\n===== 复杂度报告 =====")
    print(f"特征融合时间: {fusion_time:.4f}秒")
    print(f"波段选择总时间: {band_selection_total_time:.4f}秒")
    print(f"程序总运行时间: {total_time:.4f}秒")
    print(f"模型参数量: {total_params/1000:.2f}K")
    print(f"模型FLOPs: {flops/1e6:.2f}M")

    # 保存结果
    path = 'results/bandlist/IP/'
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f'{dataset_name}-{top_k}-250717.txt'), 'w') as f:
        f.write(str(selected_bands))
    
    return selected_bands



if __name__ == "__main__":
    main()