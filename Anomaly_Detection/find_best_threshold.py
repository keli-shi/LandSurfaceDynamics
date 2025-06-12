import numpy as np
from tqdm import tqdm
import os
import re
from osgeo import gdal


def parse_mask_name(filename):
    match = re.search(r'(h\d{2}v\d{2}).*_x(\d+)_y(\d+)', filename)
    if not match:
        raise ValueError(f"Cannot parse mask filename {filename}")
    return match.group(1), int(match.group(2)), int(match.group(3))

def parse_error_name(filename):
    match = re.search(r'(h\d{2}v\d{2})_x(\d+)_y(\d+)_abs_error', filename)
    if not match:
        raise ValueError(f"Cannot parse error filename {filename}")
    return match.group(1), int(match.group(2)), int(match.group(3))

def calculate_metrics(pred_mask, true_mask):
    pred = (pred_mask == 1)
    truth = (true_mask == 1)

    TP = np.sum(pred & truth)
    FP = np.sum(pred & (~truth))
    FN = np.sum((~pred) & truth)

    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    far = FP / (TP + FP) if (TP + FP) > 0 else 0

    return recall, far

def find_optimal_threshold_range(error_data_list, true_mask_list, recall_threshold=0.8):
    best_threshold = None
    best_far = np.inf
    best_recall = 0

    thresholds = np.arange(0, 1.0, 0.001)

    for t_low in tqdm(thresholds):
        for t_high in thresholds:
            if t_high <= t_low:
                continue

            recalls = []
            fars = []

            for error_data, true_mask in zip(error_data_list, true_mask_list):
                # print(error_data.min(), error_data.max())
                # 阈值范围二值化
                pred_mask = ((error_data >= t_low) & (error_data <= t_high)).astype(np.uint8)

                recall, far = calculate_metrics(pred_mask, true_mask)
                recalls.append(recall)
                fars.append(far)

            avg_recall = np.mean(recalls)
            avg_far = np.mean(fars)
            # print(avg_recall, avg_far)

            if avg_recall >= recall_threshold and avg_far < best_far:
                print(avg_recall, avg_far)
                best_recall = avg_recall
                best_far = avg_far
                best_threshold = (t_low, t_high)

    if best_threshold:
        print(f"✅ 最优阈值范围: [{best_threshold[0]:.4f}, {best_threshold[1]:.4f}]，平均Recall: {best_recall:.4f}，平均FAR: {best_far:.4f}")
    else:
        print("❌ 未找到满足 Recall ≥ 80% 的阈值组合")

    return best_threshold, best_far




import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Fire Mask Evaluation with Error Maps')
    parser.add_argument('--folder_mask_dir', type=str, required=True,
                        help='Path to the folder containing predicted fire masks')
    parser.add_argument('--folder_error_dir', type=str, required=True,
                        help='Path to the folder containing error maps')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("Folder with predicted masks:", args.folder_mask_dir)
    print("Folder with error maps:", args.folder_error_dir)


    mask_files = [f for f in os.listdir(args.folder_mask_dir) if f.endswith('.tif')]
    error_files = [f for f in os.listdir(args.folder_error_dir) if f.endswith('abs_error.tif')]

    error_data_list = []  # 每个为 (46, 100, 100)
    true_mask_list = []
    # 构建误差文件索引
    error_dict = {}
    for ef in error_files:
        tile, x, y = parse_error_name(ef)
        key = f"{tile}_x{x}_y{y}"
        error_dict[key] = ef

    # 遍历掩膜文件
    for mf in mask_files:
        tile, x, y = parse_mask_name(mf)
        key = f"{tile}_x{x}_y{y}"

        if key not in error_dict:
            print(f"Warning: No matching error file for {mf}")
            continue

        mask_path = os.path.join(args.folder_mask_dir, mf)
        error_path = os.path.join(args.folder_error_dir, error_dict[key])
        
        ds_mask = gdal.Open(mask_path)
        mask_data = ds_mask.ReadAsArray()  # (46,h,w)

        ds_error = gdal.Open(error_path)
        error_data = ds_error.ReadAsArray()  # (46,h,w)

        error_data_list.append(error_data)
        true_mask_list.append(mask_data)





    # 按波段逐个展开
    error_data_all = []
    true_mask_all = []

    for error_data, true_mask in zip(error_data_list, true_mask_list):
        for i in range(46):
            if np.sum(true_mask[i]) > 0:  # 只保留真值不全为0的波段
                error_data_all.append(error_data[i])
                true_mask_all.append(true_mask[i])

    # 自动搜索
    find_optimal_threshold_range(error_data_all, true_mask_all)
