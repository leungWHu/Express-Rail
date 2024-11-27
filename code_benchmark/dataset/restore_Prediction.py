# -*- coding: utf-8 -*-
# @Time    : 2024/9/3 10:37
# @Author  : lzxbetter@outlook.com
# @FileName: restore_Prediction.py
# @Software: PyCharm
""" 代码说明：
        1. 由于原始点云经过了旋转、分块。因此在与其他方法比较时，预测输出需要恢复原始点云的位置。才能更好的进行对比。
    参考：
        * 
"""
import os
import sys
import time
import math
import numpy as np

from My_utils.las import write_las_fit

# rotate_dict_path = "/home/gisleung/桌面/新建文件夹/结果对比图-SNCF/rotate_dict-202409-1316.pkl"
# prediction_path = "/home/gisleung/桌面/新建文件夹/结果对比图-SNCF/KPConv/"
# prediction_fields = ['x', 'y', 'z', 'gt', 'diff', 'preds']

rotate_dict_path = "/home/gisleung/桌面/新建文件夹/结果对比图-WHU/rotate_dict-202409-1009.pkl"
prediction_path = "/home/gisleung/桌面/新建文件夹/结果对比图-WHU/kpconv/"
prediction_fields = ['x', 'y', 'z', 'gt', 'diff', 'preds']

# 1. 读取 prediction_path 下的所有文件夹名称
prediction_folders = os.listdir(prediction_path)
prediction_folders = [folder for folder in prediction_folders if os.path.isdir(os.path.join(prediction_path, folder))]
prediction_folders.sort()
print(f"prediction_folders: {prediction_folders}")

# 2. 读取 rotate_dict
import pickle
with open(rotate_dict_path, 'rb') as f:
    rotate_dict = pickle.load(f)
# print(f"rotate_dict: {rotate_dict}")


# 3. 读取每个文件夹下的 txt 文件，合并（使用进度条）
from tqdm import tqdm

for folder in tqdm(prediction_folders):
    file_data = []
    temp_dir = os.path.join(prediction_path, folder, 'visual')
    # 读取目录下所有的 txt 文件，并合并为一个 numpy 数组
    all_txt_files = os.listdir(temp_dir)
    all_txt_files = [file for file in all_txt_files if file.endswith('.txt')]
    all_txt_files.sort()
    # 读取每个 txt 文件
    for txt_file in all_txt_files:
        txt_path = os.path.join(temp_dir, txt_file)
        data = np.loadtxt(txt_path)
        file_data.append(data)

    # 拼接
    data = np.concatenate(file_data, axis=0)
    print(f"folder: {folder}, data.shape: {data.shape}")

    # 4. 旋转点云
    rotation_matrix = rotate_dict[folder]
    # 第一次旋转 rotated_points = np.dot(pc_array[:, 0:2], rotation_matrix.T)
    # 逆向旋转，恢复原始点云
    rotated_points = np.dot(data[:, 0:2], rotation_matrix)
    data[:, 0:2] = rotated_points
    xyz = data[:, 0:3]

    write_las_fit(os.path.join(prediction_path, folder, f'{folder}.las'), xyz, None, {
        prediction_fields[3]: data[:, 3],
        prediction_fields[4]: data[:, 4],
        prediction_fields[5]: data[:, 5]
    })




