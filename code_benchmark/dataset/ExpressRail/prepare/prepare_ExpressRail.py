# -*- coding: utf-8 -*-
# @Time    : 2024/4/12 下午8:10
# @Author  : lzxbetter@outlook.com
# @FileName: prepare_Leung.py
# @Software: PyCharm
""" 代码说明：
        更新日期：2024/9/3

        预处理功能：将铁路点云进行分块，分块逻辑：
        1. 将点云旋转至水平，使得铁轨方向平行于X轴
        2. 将点云进行下采样至0.05m
        3. 将点云分割成多个块，根据预设参数，分别沿着X轴和Y轴分割，记录每一块的起止点
        4. 根据起止点，分割原始点云（旋转后的），得到分块数据
        5. 存储分块数据，同时存储KD树，用于后续的查询
            * 训练集，只存储分块（下采样的）点云、标签、特征、KD树
            * 测试集，存储分块（下采样的）点云、标签、特征、KD树；原始点云标签、原始点云对应的下采样点云索引

        （9月3日更新）
        1. 记录每个文件的旋转角度，并输出txt
        2.

    参考：
        在DealCfg中设置参数，运行 python prepare_Leung2.py
        当
            # static_length = [1, 0.5]  # 用于微分统计的分辨率
            # split_val = [15, 100000]  # 用于分割的分辨率
            # split_key = ['length', 'number']  # 分割的单位
        时，先进行横向分割，再进行纵向分割；

        当
            # static_length = [1]  # 用于微分统计的分辨率
            # split_val = [15]  # 用于分割的分辨率
            # split_key = ['length']  # 分割的单位
        时，只进行横向分割
"""
import os
import pickle
import sys
import time
import math

import laspy
import numpy as np
from plyfile import PlyData
import pandas as pd
from tqdm import tqdm
import multiprocessing
from sklearn.neighbors import KDTree

# from ml3d.datasets.utils import DataProcessing
from open3d._ml3d.datasets.utils import DataProcessing


# from utils.data_process import DataProcessing as DP

def create_kd_tree(pc):
    """
    生成kd树
    :param pc:
    :return:
    """
    from sklearn.neighbors import KDTree
    kdt = KDTree(pc)
    return kdt


def create_dirs(root, file):
    file_without_suffix = file.split('.')[0]
    date_save = os.path.join(root, file_without_suffix)
    os.makedirs(os.path.join(date_save, 'cache'), exist_ok=True)

    # 如果是检查数据
    if file in DealCfg.check_files:
        os.makedirs(os.path.join(date_save, 'block_org'), exist_ok=True)
        os.makedirs(os.path.join(date_save, 'block'), exist_ok=True)


def read_ply_use_plyfile(pc_path):
    """
    返回xyz和所有有效属性值
    """
    plydata = PlyData.read(pc_path)  # 读取ply文件
    data = plydata.elements[0].data  # 读取数据
    data_pd = pd.DataFrame(data)  # 转为 DataFrame
    pc_array = np.zeros(data_pd.shape, dtype=np.float64)  # initialize array to store data
    property_names = data[0].dtype.names  # read names of properties
    for i, name in enumerate(property_names):  # read data by property
        pc_array[:, i] = data_pd[name]
        print(f"第 {i + 1} 个属性值是 {name}")
    return pc_array.astype(np.float32)


def rotate_to_horizontal(pc_array, labelIndex=None, byLabel=None):
    """
    旋转点云至水平位置(前提是点云沿着铁轨方向前进)
    :param pc_array: 点云读取结果 n * f
    :param labelIndex: 标签所在的列
    :param byLabel: 旋转时以哪个标签为准
    :return:
    """

    # 先取出依据的 xoy 点
    pc_xoy = pc_array[:, 0:2]
    if labelIndex and byLabel:
        label = pc_array[:, labelIndex:labelIndex + 1]
        rail_colum = np.where(label == byLabel)[0]
        pc_xoy = pc_array[:, 0:2][rail_colum, :]

    # 随机减少点的个数
    sample_num = 10000
    if pc_xoy.shape[0] > sample_num:
        random_indices = np.random.choice(pc_xoy.shape[0], size=sample_num, replace=False)
        pc_xoy = pc_xoy[random_indices]

    # 计算旋转角度
    # 计算点云的协方差矩阵，并求解该矩阵的特征向量：
    cov_mat = np.cov(np.transpose(pc_xoy))
    eigen_values, eigen_vectors = np.linalg.eig(cov_mat)
    # 找到最大特征值对应的特征向量，即为点云在该方向下的投影长度最大的方向向量：
    max_eigen_value_index = np.argmax(eigen_values)
    max_eigen_vector = eigen_vectors[:, max_eigen_value_index]
    # 计算旋转角度
    angle = -np.arctan2(max_eigen_vector[1], max_eigen_vector[0])
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    # print("旋转角度：", angle * 180 / np.pi)
    # 进行旋转
    rotated_points = np.dot(pc_array[:, 0:2], rotation_matrix.T)
    pc_array[:, :2] = rotated_points
    return pc_array, rotation_matrix


def split_pc_flexible(pc, static_length=1, split_val=15, split_key='length', switch_XY=False, print_flag=True):
    """

    Args:
        pc:
        static_length:
        split_val:
        split_key:
        switch_XY:
        print_flag:

    Returns:
        segment_pc_list # 每一段点云
        block_start_end_list # 每一段点云的起止点

    """

    pc_read = pc.copy()

    # 是否交换xy
    if switch_XY:
        pc_read[:, [0, 1]] = pc_read[:, [1, 0]]

    # x轴的范围
    x_min = np.min(pc_read[:, 0])
    x_amx = np.max(pc_read[:, 0])

    # 在x轴的两端，分别取20m的点云，判断点个数大小。如果终点点数大于起点点数，则原始点云x坐标取负
    reverse_X_flag = False
    x_min_pc = pc_read[(pc_read[:, 0] >= x_min) & (pc_read[:, 0] <= x_min + 20)]
    x_max_pc = pc_read[(pc_read[:, 0] >= x_amx - 20) & (pc_read[:, 0] <= x_amx)]
    if print_flag:
        print(f"起点点数：{x_min_pc.shape[0]}")
        print(f"终点点数：{x_max_pc.shape[0]}")
    if x_min_pc.shape[0] < x_max_pc.shape[0]:
        reverse_X_flag = True
        pc_read[:, 0] = -pc_read[:, 0]
        x_min = np.min(pc_read[:, 0])
        x_amx = np.max(pc_read[:, 0])

    # 每2m分段统计（可以认为是重叠度）
    segment_length = static_length
    segment_number = math.ceil((x_amx - x_min) / segment_length)
    if print_flag:
        print(f"分割段数：{segment_number}")
    # 每一段的长度和起止点
    segment_start_end_list = []
    segment_length_list = []
    for i in range(segment_number):
        segment_start = x_min + i * segment_length
        segment_end = x_min + (i + 1) * segment_length
        segment_start_end_list.append([segment_start, segment_end])
        segment_length_list.append(segment_length)
    segment_start_end_list[-1][1] = x_amx + 1
    segment_length_list[-1] = segment_start_end_list[-1][1] - segment_start_end_list[-1][0]

    # 分割点云，获取每一段点云的索引
    segment_index_list = []
    for i in range(segment_number):
        segment_index = np.where((pc_read[:, 0] >= segment_start_end_list[i][0]) &
                                 (pc_read[:, 0] < segment_start_end_list[i][1]))
        segment_index_list.append(segment_index[0])

    # 统计每一段点云的点数
    segment_point_number_list = []
    for idxs in segment_index_list:
        segment_point_number_list.append(len(idxs))
    if print_flag:
        print(f"每一段点云的点数：{segment_point_number_list}")
        # 平均值
        print(f"平均值：{np.mean(segment_point_number_list)}")
        print(f"共：{sum(segment_point_number_list)}")
        print()

    # 对数量排序，计算前90%的平均值
    temp_copy = segment_point_number_list.copy()
    temp_copy.sort(reverse=True)
    temp_copy = np.array(temp_copy)
    percent_90 = temp_copy[0:math.ceil(segment_number * 0.9)]
    percent_90_mean = np.mean(percent_90)
    if print_flag:
        print(f"前90%的平均值：{percent_90_mean}")

    if split_key == 'length':
        # 每一段的长度
        part_length = split_val
        segment_part_number = int(percent_90_mean * part_length / segment_length)
        if print_flag:
            print(f"按照{part_length}m分段分割，每个段应包含点数：{segment_part_number}")
            print()
    elif split_key == 'number':
        segment_part_number = split_val
        if print_flag:
            print(f"直接按点数进行估计分段，每个段包含点数：{segment_part_number}")
            print()

    # 根据segment_point_number_list进行连续分组，每个分组的点数之和应该大于segment_part_number
    group_list = []
    group_num_list = []

    group_item = []
    group_num_item = 0

    loop_flag = True
    cur_index = 0  # segment_point_number_list

    while loop_flag:
        if group_num_item < segment_part_number:
            group_item.append(cur_index)
            group_num_item += segment_point_number_list[cur_index]
            cur_index += 1
            if cur_index == len(segment_point_number_list):
                group_list.append(group_item)
                group_num_list.append(group_num_item)
                loop_flag = False
        else:
            group_list.append(group_item)
            group_num_list.append(group_num_item)
            group_item = []
            group_num_item = 0
            # cur_index -= 1  #重叠
    if print_flag:
        print(f"分组结果：{group_list}")
        print(f"分组数量：{group_num_list}, \n分组后总数：{sum(group_num_list)}")

    """ 对最后一个分组做处理 """
    # 如果最后一个分组的点数量小于segment_part_number的2/4，则将最后两个分组合并
    if len(group_num_list) > 1 and (group_num_list[-1] < segment_part_number / 2):
        if print_flag:
            print("-----> 最后一个分组的点数量小于segment_part_number的2/4")
        group_list[-2].extend(group_list[-1])
        group_num_list[-2] += group_num_list[-1]
        group_list.pop()
        group_num_list.pop()

        if print_flag:
            print(f"新-分组结果：{group_list}")
            print(f"新-分组数量：{group_num_list}, \n总数：{sum(group_num_list)}")
            print()

    """ 新增部分：根据分组结果，计算每一block点云的起止点 """
    block_start_end_list = []
    for group in group_list:
        start = segment_start_end_list[group[0]][0]
        end = segment_start_end_list[group[-1]][1]
        block_start_end_list.append([start, end])  # 前闭后开 [start, end)
    if reverse_X_flag:
        # 掉转list
        block_start_end_list.reverse()
        for item in block_start_end_list:
            item[0] = -item[0]
            item[1] = -item[1]
            item.reverse()
    block_start_end_list[0][0] = block_start_end_list[0][0] - 2
    block_start_end_list[-1][1] = block_start_end_list[-1][1] + 2

    # 根据分组结果，将点云分割成多段
    segment_index_list_list = []
    for group in group_list:
        segment_index_list_item = []
        for idx in group:
            segment_index_list_item.append(segment_index_list[idx])
        segment_index_list_list.append(segment_index_list_item)
    # print(f"分组后的索引：{segment_index_list_list}")

    # 根据segment_index_list_list 提取出每一段的点云
    segment_pc_list = []
    segment_pc_number_list = []
    for segment_index_list_item in segment_index_list_list:
        # 将segment_index_list_item中所有的ndarray，合并为一个ndarray
        segment_index_list_item = np.concatenate(segment_index_list_item)
        segment_pc = pc_read[segment_index_list_item]

        if reverse_X_flag:
            segment_pc[:, 0] = -segment_pc[:, 0]
        if switch_XY:
            segment_pc[:, [0, 1]] = segment_pc[:, [1, 0]]
        segment_pc_list.append(segment_pc)
        segment_pc_number_list.append(len(segment_pc))

        if reverse_X_flag:
            segment_pc_list.reverse()
            segment_pc_number_list.reverse()

    if print_flag:
        # print(f"分组后的点云：{segment_pc_list}")
        print(
            f"分为{len(segment_pc_number_list)}块：{segment_pc_number_list}, 总数：{sum(segment_pc_number_list)}, 平均：{np.mean(segment_pc_number_list)}")
    return segment_pc_list, block_start_end_list


def split_pc(pc_array, static_length=[1, 0.5], split_val=[10, 100000], split_key=['length', 'number'],
             switch_XY=[False, True], print_flag=[False, False]):
    """

    :param pc_array: 经过旋转且下采样的点云 （且沿着X轴延伸）
    :param static_length: 统计分辨率（分辨率约小，分割结果越接近期望指标split_val）
    :param split_val: 分割分辨率
    :param split_key: 分割分辨率的单位
    :param switch_XY: 置换XY轴 (沿着铁轨方向切割，不需要置换；沿着铁轨垂直方向切割，需要置换)
    :param print_flag: 打印过程
    :return:
        result  # 分割后的点云
        sub_limit  # 第一次分割的分割点
        sec_limits  # 第二次分割的分割点
    """

    result = []

    # 第一次分割，沿着X轴，等间距分割
    blocks, sub_limit = split_pc_flexible(pc_array, static_length[0], split_val[0],
                                          split_key[0], switch_XY[0], print_flag[0])

    # 沿着X轴分割
    if len(split_val) == 1:
        result = blocks
        sec_limits = [[[-10000.0, 10000.0]] for _ in range(len(blocks))]  # 缺省值
        return result, sub_limit, sec_limits

    # 在上一步分割基础上，沿着Y轴分割
    elif len(split_val) == 2:
        sec_limits = []
        for i, block in enumerate(blocks):
            # 第二次分割，沿着Y轴，等点数分割
            sub_points, blocks_limit = split_pc_flexible(block, static_length[1], split_val[1],
                                                         split_key[1], switch_XY[1], print_flag[1])
            result.append(sub_points)
            sec_limits.append(blocks_limit)
        return result, sub_limit, sec_limits


def grid_down_sample(points, labels, feat=None, grid_size=0.03):
    """ 下采样至0.03 """
    if feat is None:
        sub_points, sub_labels = DataProcessing.grid_subsampling(
            points, labels=labels, grid_size=DealCfg.grid_size)
        sub_feat = None
        # 拼接ub_points, sub_labels
        # sub_pc = np.hstack((sub_points, sub_labels.reshape(-1, 1)))
    else:
        sub_points, sub_feat, sub_labels = DataProcessing.grid_subsampling(
            points, features=feat, labels=labels, grid_size=DealCfg.grid_size)
        # 拼接ub_points, sub_labels, sub_feat
        # sub_pc = np.hstack((sub_points, sub_labels.reshape(-1, 1), sub_feat))

    return sub_points, sub_labels, sub_feat


def MultiProcessFun(file_path):
    file = file_path.split('/')[-1]  # fileName
    data = {}
    # 创建一个文件夹，用于存储分割后的点云
    create_dirs(DealCfg.save_dir, file)
    """ 读取点云数据，并旋转至水平 """
    las_read = laspy.read(file_path)
    las_label = las_read.label.astype(np.int32)
    las_xyz = las_read.xyz.astype(np.float32)
    pc_read = np.hstack((las_xyz, las_label.reshape(-1, 1)))

    pc_rotate_label, rotate_m = rotate_to_horizontal(pc_read)
    DealCfg.rotate_dict[file.split('.')[0]] = rotate_m
    # print()
    # print('》》》》临时截断，直接return')
    # return

    """ 下采样至0.03 """
    points = pc_rotate_label[:, 0:3].astype(np.float32)
    feat = None
    labels = pc_rotate_label[:, 3]
    # 转为int32
    labels = labels.astype(np.int32)

    if feat is None:
        sub_points, sub_labels = DataProcessing.grid_subsampling(
            points, labels=labels, grid_size=DealCfg.grid_size)
        sub_feat = None
        # 拼接ub_points, sub_labels
        sub_pc = np.hstack((sub_points, sub_labels.reshape(-1, 1)))
    else:
        sub_points, sub_feat, sub_labels = DataProcessing.grid_subsampling(
            points, features=feat, labels=labels, grid_size=DealCfg.grid_size)
        # 拼接ub_points, sub_labels, sub_feat
        sub_pc = np.hstack((sub_points, sub_labels.reshape(-1, 1), sub_feat))

    # print()

    """ 分块 """
    """0.03"""
    # all_blocks, sub_limits, sec_limits = split_pc(sub_pc, [1, 0.5], [10, 100000],
    #                                               ['length', 'number'], [False, True], [False, False])
    """0.05"""
    all_blocks, sub_limits, sec_limits = split_pc(sub_pc,
                                                  DealCfg.static_length, DealCfg.split_val,
                                                  ['length', 'number'], [False, True], [False, False])

    # 遍历二维list(存了每个很快的边界)，并保存
    with tqdm(total=len(sub_limits), desc=f'子进度 > {file} 处理进度') as bar:  # total表示预期的迭代次数
        for i, sub_limit in enumerate(sub_limits):
            # 先切一刀
            sub_point_label = pc_read[(pc_read[:, 0] >= sub_limit[0]) & (pc_read[:, 0] < sub_limit[1])]
            for j, sec_limit in enumerate(sec_limits[i]):
                block = sub_point_label[
                    (sub_point_label[:, 1] >= sec_limit[0]) & (sub_point_label[:, 1] < sec_limit[1])]
                if block.size == 0:
                    print(f" > > > > ! ! ! ! < < < < 0个点： {file.split('.')[0]}_{i}_{j}")

                # 开始将采样、KD树等预处理操作
                points = block[:, 0:3].astype(np.float32)
                labels = block[:, 3]
                labels = labels.astype(np.int32)  # 转为int32
                sub_points, sub_labels, sub_feat = grid_down_sample(points, labels, None, grid_size=DealCfg.grid_size)

                search_tree = KDTree(sub_points)  # 构建KD树

                data['point'] = sub_points
                data['feat'] = sub_feat
                data['label'] = sub_labels
                data['search_tree'] = search_tree

                """ 如果是测试集, （因为要映射为原始全部点，所以记录proj_inds）"""
                if file in DealCfg.test_files:
                    proj_inds = np.squeeze(
                        search_tree.query(points, return_distance=False))  # 原始数量124668  下采样83891
                    proj_inds = proj_inds.astype(np.int32)
                    data['proj_inds'] = proj_inds
                    data['ori_point_label'] = block[:, 0:4]

                file_name = f"{file.split('.')[0]}_{i}_{j}"
                # print(" > file_name: ", file_name)
                fpath = os.path.join(DealCfg.save_dir, file.split('.')[0], "cache", str('{}.npy'.format(file_name)))
                # # 存储数据
                np.save(fpath, data)

                """ 如果需要检查数据 """
                # if file in list(set(DealCfg.check_files + DealCfg.test_files)):
                # if file in DealCfg.test_files:
                if file in DealCfg.check_files:
                    org_block = block
                    np.savetxt(
                        os.path.join(DealCfg.save_dir, file.split('.')[0], "block_org", f'{file_name}.txt'), org_block)
                    # 如果是检查数据，同时存储下采样的数据
                    if file in DealCfg.check_files:
                        sub_block = np.hstack((sub_points, sub_labels.reshape(-1, 1)))
                        np.savetxt(
                            os.path.join(DealCfg.save_dir, file.split('.')[0], "block", f'{file_name}.txt'), sub_block)

            bar.update(1)  # 每次更新进度条的长度


class DealCfg:
    ''' 重要的分割参数 '''
    grid_size = 0.05

    static_length = [1, 0.5]  # 用于微分统计的分辨率
    split_val = [15, 150000]  # 用于分割的分辨率
    split_key = ['length', 'number']  # 分割的单位

    # static_length = [1]  # 用于微分统计的分辨率
    # split_val = [15]  # 用于分割的分辨率
    # split_key = ['length']  # 分割的单位

    root_dir = "data/Express-Rail"
    pc_dir = os.path.join(root_dir, "V1.0")
    save_dir = os.path.join(root_dir, 'Open3d_deal', f"grid_{grid_size}/")
    num_classes = 9

    # 测试集的数据列表
    train_file_names = os.path.join(root_dir, "V1.0/train_file_names.txt")
    test_file_names = os.path.join(root_dir, "V1.0/test_file_names.txt")
    # 检查文件
    check_files = ["L04-label-2.las"]

    with open(train_file_names, 'r') as file:
        train_files = [f"{line.strip()}" for line in file]

    with open(test_file_names, 'r') as file:
        test_files = [f"{line.strip()}" for line in file]

    train_files.sort()
    test_files.sort()

    count_train = np.zeros(num_classes)
    count_test = np.zeros(num_classes)

    time_str = time.strftime("%Y%m-%d%H", time.localtime())
    rotate_dict = {}  # 每个文件的旋转矩阵


if __name__ == '__main__':
    print('go!')

    # 设置更目录为工作目录
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 设置随机数种子为固定值，比如0
    np.random.seed(7777)

    # print(f" >>>> 测试用 ... <<< ")
    # 从第20个开始截取list: files
    # files = files[16:]  # L14-1-M01-003
    # files = ['L7-1-M01-001.ply']  # L14-1-M01-003

    # 在此处修改文件列表，做临时debug用
    # print(f" >>>> 测试用 ... <<< ")
    # DealCfg.train_files = ['L04-label-2.las']
    # DealCfg.test_files = ['L04-label-4.las']

    # 写入完整路径
    files_train = [os.path.join(DealCfg.pc_dir, 'train', file) for file in DealCfg.train_files]
    files_test = [os.path.join(DealCfg.pc_dir, 'test', file) for file in DealCfg.test_files]
    # 合并
    files = files_train + files_test
    files.sort()  # files 排序
    print("获取路径下的所有文件：", files)

    """ 正式程序： """
    pbar = tqdm(files)
    for file in pbar:
        pbar.set_description("总进度 > 正在处理 %s" % file)  # ("当前正在处理 %s" % file)
        MultiProcessFun(file)
    pbar.close()

    ## ## ## 记录 ## ## ##
    """ 将旋转矩阵，记录到 save_dir 下的 rotate_dict.np 文件中 """
    save_rotate = os.path.join(DealCfg.save_dir, f'rotate_dict-{DealCfg.time_str}.pkl')
    with open(save_rotate, 'wb') as f:
        pickle.dump(DealCfg.rotate_dict, f)
    print("旋转矩阵已保存！")
    # 读取恢复
    # with open(save_rotate, 'rb') as f:
    #     data = pickle.load(f)

    """ 将Config中的参数，记录到 save_dir 下的 config.txt 文件中 """
    with open(os.path.join(DealCfg.save_dir, f'config-{DealCfg.time_str}.txt'), 'w') as f:
        f.write(f"grid_size: {DealCfg.grid_size}\n")
        f.write(f"static_length: {DealCfg.static_length}\n")
        f.write(f"split_val: {DealCfg.split_val}\n")
        f.write(f"split_key: {DealCfg.split_key}\n")
        f.write(f"train_files: {DealCfg.train_files}\n")
        f.write(f"test_files: {DealCfg.test_files}\n")
        f.write(f"check_files: {DealCfg.check_files}\n")
        f.write(f"num_classes: {DealCfg.num_classes}\n")

    """ 统计标签权重，并记录在文件 label_static.txt 中 """
    print("> > > 开始统计标签数量：")
    for file_path in files:
        file = file_path.split('/')[-1]  # fileName
        sub_dir = os.path.join(DealCfg.save_dir, file.split('.')[0], "cache")
        sub_files = os.listdir(sub_dir)
        for sub_file in sub_files:
            data = np.load(os.path.join(sub_dir, sub_file), allow_pickle=True).item()
            labels = data['label']
            count = np.bincount(labels, minlength=DealCfg.num_classes)
            if file in DealCfg.train_files:
                DealCfg.count_train += count
            elif file in DealCfg.test_files:
                DealCfg.count_test += count

    # 转为 list 并打印
    print("训练集标签数量：", DealCfg.count_train.tolist())
    print("测试集标签数量：", DealCfg.count_test.tolist())
    print(" ! ! ! > 请 copy 至 config_***.yaml < ! ! ! ")

    # 将标签数量，记录到 save_dir 下的 label_static.txt 文件中
    with open(os.path.join(DealCfg.save_dir, f'label_static-{DealCfg.time_str}.txt'), 'w') as f:
        f.write(f"count_train: {DealCfg.count_train.tolist()}\n")
        f.write(f"count_test: {DealCfg.count_test.tolist()}\n")
