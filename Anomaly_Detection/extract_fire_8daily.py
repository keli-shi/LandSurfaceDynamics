from osgeo import gdal
gdal.UseExceptions()

import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os



def binarize(data, threshold=0.1):
    """
    将 abs_error 二值化：
    小于等于 threshold -> 0
    大于 threshold -> 1
    """
    binary_data = (data > threshold).astype(np.uint8)  # 使用 uint8 (0, 1)
    return binary_data

def hdf_to_multiband_tif(hdf_list, output_tif, subdataset_index=0):
    """
    将多个 MODIS HDF 文件的指定子数据集读取，并合并为一个多波段 GeoTIFF。
    
    :param hdf_list: HDF 文件路径列表
    :param output_tif: 输出多波段 TIF 文件路径
    :param subdataset_index: 使用的子数据集索引，默认使用第一个（可用 gdalinfo 查看）
    """
    # 读取第一个hdf，打开指定子数据集作为模板
    print(hdf_list[0])
    hdf_example = gdal.Open(hdf_list[0])
    subdatasets = hdf_example.GetSubDatasets()
    subdataset_path = subdatasets[subdataset_index][0]
    print(subdataset_path)
    ref_ds = gdal.Open(subdataset_path)
    
    x_size = ref_ds.RasterXSize
    y_size = ref_ds.RasterYSize
    projection = ref_ds.GetProjection()
    geotransform = ref_ds.GetGeoTransform()
    
    # 创建输出多波段TIF
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_tif, x_size, y_size, len(hdf_list), gdal.GDT_Byte)
    
    # 遍历每个HDF，读取指定子数据集，写入对应波段
    for idx, hdf_file in enumerate(hdf_list):
        hdf_ds = gdal.Open(hdf_file)
        subdatasets = hdf_ds.GetSubDatasets()
        
        # 获取目标子数据集
        subdataset_path = subdatasets[subdataset_index][0]
        subdataset = gdal.Open(subdataset_path)
        
        band_data = subdataset.GetRasterBand(1).ReadAsArray()
        print(band_data.shape)
        print(band_data.min(), band_data.max())

        band_data = binarize(band_data, threshold=6)
        print(band_data.sum())

        out_band = out_ds.GetRasterBand(idx + 1)   
        out_band.WriteArray(band_data) 
        
        print(f'Band {idx+1} from {os.path.basename(hdf_file)} added.') 

    # 设置空间参考
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    
    out_ds.FlushCache()
    out_ds = None
    
    print(f'Multi-band TIF saved: {output_tif}')



import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Process directories for FireMask generation')
    parser.add_argument('--input_dir', type=str, default='Data',
                        help='Path to the input data directory')
    parser.add_argument('--output_dir', type=str, default='2023年8天tif_FireMask',
                        help='Path to save output files')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("Input directory:", args.input_dir)
    print("Output directory:", args.output_dir)

    for hv in os.listdir(args.input_dir):

        hv_dir_path = os.path.join(args.input_dir, hv)
        hdf_files = [os.path.join(hv_dir_path,f) for f in os.listdir(hv_dir_path) if f.endswith('.hdf')]
        hdf_files.sort()

        hdf_to_multiband_tif(hdf_files[-46:], output_tif=f'{args.output_dir}/{hv}_2023_8daily.tif', subdataset_index=0)