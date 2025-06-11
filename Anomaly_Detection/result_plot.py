import os
import re
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def parse_mask_name(filename):
    match = re.search(r'(h\d{2}v\d{2}).*_x(\d+)_y(\d+)', filename)
    if not match:
        raise ValueError(f"Cannot parse mask filename {filename}")
    return match.group(1), int(match.group(2)), int(match.group(3))

def parse_error_name(filename):
    match = re.search(r'_(h\d{2}v\d{2})_.*?_x(\d+)_y(\d+)_abs_error', filename)
    if not match:
        raise ValueError(f"Cannot parse error filename {filename}")
    return match.group(1), int(match.group(2)), int(match.group(3))

def plot_prediction_vs_truth_side_by_side_with_stats(pred_mask, true_mask, title_prefix, save_path=None):
    bands, h, w = pred_mask.shape

    # 找到真值不全为0的波段
    valid_band_indices = [i for i in range(bands) if np.sum(true_mask[i]) > 0]

    if len(valid_band_indices) == 0:
        print(f"Warning: {title_prefix} 全部波段真值为空，不绘制")
        return

    # 2行，列数自动
    ncols = int(np.ceil(len(valid_band_indices) / 2))
    nrows = 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()

    cmap_custom = ListedColormap(['lightgrey', 'green'])

    for idx, band_idx in enumerate(valid_band_indices):
        ax = axes[idx]

        pred = pred_mask[band_idx]
        truth = true_mask[band_idx]

        # 组合图像（左预测，右真值）
        combined = np.zeros((h, w * 2), dtype=np.uint8)
        combined[:, :w] = pred
        combined[:, w:] = truth

        pred_count = np.sum(pred)
        truth_count = np.sum(truth)

        # 计算检测率和虚警率
        TP = np.sum((pred == 1) & (truth == 1))
        FP = np.sum((pred == 1) & (truth == 0))

        recall = TP / truth_count if truth_count > 0 else 0
        false_alarm_rate = FP / pred_count if pred_count > 0 else 0

        ax.imshow(combined, cmap=cmap_custom, vmin=0, vmax=1)
        ax.set_title(
            f'Band {band_idx+1}\nPred: {pred_count} | Truth: {truth_count}\nRecall: {recall:.2f} | FAR: {false_alarm_rate:.2f}',
            fontsize=8
        )
        ax.axis('off')

    # 隐藏多余子图
    for j in range(len(valid_band_indices), len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"{title_prefix} - Prediction (Left) vs Truth (Right)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300)
    # plt.show()
    plt.close()



def process_folders(folder_mask, folder_error, threshold_low=0, threshold_high=0.0986, save_folder=None):
    os.makedirs(save_folder, exist_ok=True) if save_folder else None

    mask_files = [f for f in os.listdir(folder_mask) if f.endswith('.tif')]
    error_files = [f for f in os.listdir(folder_error) if f.endswith('abs_error.tif')]

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

        mask_path = os.path.join(folder_mask, mf)
        error_path = os.path.join(folder_error, error_dict[key])

        ds_mask = gdal.Open(mask_path)
        mask_data = ds_mask.ReadAsArray()  # (46,h,w)

        ds_error = gdal.Open(error_path)
        error_data = ds_error.ReadAsArray()  # (46,h,w)

        # 预测掩膜
        pred_mask = ((error_data >= threshold_low) & (error_data <= threshold_high)).astype(np.uint8)

        # 绘制
        title_prefix = f"{key}"
        save_name = os.path.join(save_folder, f"{key}_compare.png") if save_folder else None
        plot_prediction_vs_truth_side_by_side_with_stats(pred_mask, mask_data, title_prefix, save_path=save_name)

        ds_mask = None
        ds_error = None

    print("All files processed and plotted.")


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Fire Mask Detection Evaluation')

    parser.add_argument('--folder_mask_dir', type=str, required=True,
                        help='Directory containing predicted fire mask TIFFs')
    parser.add_argument('--folder_error_dir', type=str, required=True,
                        help='Directory containing error map TIFFs')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save the output results')
    parser.add_argument('--threshold_low', type=float, required=True,
                        help='Lower threshold value for fire detection')
    parser.add_argument('--threshold_high', type=float, required=True,
                        help='Upper threshold value for fire detection')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    print("Predicted Mask Directory:", args.folder_mask_dir)
    print("Error Map Directory:", args.folder_error_dir)
    print("Output Save Directory:", args.save_dir)
    print("Threshold Low:", args.threshold_low)
    print("Threshold High:", args.threshold_high)


    process_folders(args.folder_mask_dir, args.folder_error_dir, threshold_low=args.threshold_low, threshold_high=args.threshold_high, save_folder=args.save_dir)
