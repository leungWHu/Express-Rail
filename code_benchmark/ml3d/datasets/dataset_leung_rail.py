import time
from datetime import datetime

import laspy
import numpy as np
import os, argparse, pickle, sys
from os.path import exists, join, isfile, dirname, abspath, split
import logging

from sklearn.neighbors import KDTree
import yaml
from tqdm import tqdm

from .base_dataset import BaseDataset, BaseDatasetSplit
from .utils import DataProcessing
from ..utils import make_dir, DATASET

log = logging.getLogger(__name__)



def create_new_las(out_file, xyz, labels=None, sourceId=None, gps_time=None):
    # 1. Create a new header
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.add_extra_dim(laspy.ExtraBytesParams(name="label", type=np.int32))

    # 2. Create a Las
    las = laspy.LasData(header)
    # 添加xyz坐标
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    # 添加GPS时间
    if gps_time is not None:
        las.gps_time = gps_time
    # 添加标签
    if labels is not None:
        las.label = labels
    # 添加sourceId
    if sourceId is not None:
        las.point_source_id = sourceId

    # 保存LAS文件
    las.write(out_file)

class LeungRail(BaseDataset):
    """This class is used to create a dataset based on the SemanticKitti
    dataset, and used in visualizer, training, or testing.

    The dataset is best for semantic scene understanding.
    """

    def __init__(self,
                 dataset_path,
                 info,
                 name='LeungRail',
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            name: The name of the dataset (Semantic3D in this case).
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.
            class_weights: The class weights to use in the dataset.
            ignored_label_inds: A list of labels that should be ignored in the dataset.
            test_result_folder: The folder where the test results should be stored.

        Returns:
            class: The corresponding class.
        """
        super().__init__(dataset_path=dataset_path, name=name, **kwargs)

        cfg = self.cfg

        self.info = info
        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)

    def get_label_to_names(self):
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        # 把 self.info中的 label 和 names 转为 dict
        label_to_names = {self.info["label"][i]: self.info["names"][i] for i in range(len(self.info["names"]))}
        return label_to_names

    def get_split(self, split):
        """Returns a dataset split.
        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return LeungRailSplit(self, split=split)

    def is_tested(self, attr):
        """Checks if a datum in the dataset has been tested.

        Args:
            attr: The attribute that needs to be checked.

        Returns:
            If the datum attribute is tested, then return the path where the
                attribute is stored; else, returns false.
        """
        cfg = self.cfg
        name = attr['name']
        name_seq, name_points = name.split("_")
        test_path = join(cfg.test_result_folder, 'sequences')
        save_path = join(test_path, name_seq, 'predictions')
        test_file_name = name_points
        store_path = join(save_path, name_points + '.label')
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    def save_test_result(self, results, attr, ori_point_label=None):
        """Saves the output of a model.

        Args:
            ori_point_label:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
            ori_point: The original point cloud data.
        """
        cfg = self.cfg
        name = attr['name']
        name_seq, test_file_name = name.split("#")

        # test_path = join(cfg.test_result_folder, 'sequences')
        save_path = join(cfg.test_result_folder, name_seq, 'predictions')
        make_dir(save_path)

        pred = results['predict_labels']

        # TODO: 这里的 ignored_label_inds 需要根据实际处理
        # for ign in cfg.ignored_label_inds:
        #     pred[pred >= ign] += 1

        store_path = join(save_path, test_file_name + '.label')
        # pred = self.remap_lut[pred].astype(np.uint32)
        pred.tofile(store_path)

        # 存储预测结果的可视化
        if ori_point_label is not None:
            ori_label = ori_point_label[:, -1:]
            pre_label = pred.reshape(-1, 1)
            diff = pre_label - ori_label
            # diff 不等于0的值，说明预测错误，标记为1
            diff[diff != 0] = 1
            out_data = np.concatenate([ori_point_label[:, 0:3], ori_label, diff, pre_label], axis=1)
            # 保存为txt文件
            save_path = join(cfg.test_result_folder, name_seq, 'visual')
            make_dir(save_path)
            store_file = join(save_path, test_file_name + '.txt')
            np.savetxt(store_file, out_data)

    def save_test_result_kpconv(self, results, inputs):
        cfg = self.cfg
        for j in range(1):
            name = inputs['attr']['name']
            name_seq, name_points = name.split("_")

            test_path = join(cfg.test_result_folder, 'sequences')
            make_dir(test_path)
            save_path = join(test_path, name_seq, 'predictions')
            make_dir(save_path)

            test_file_name = name_points

            proj_inds = inputs['data'].reproj_inds[0]
            probs = results[proj_inds, :]

            pred = np.argmax(probs, 1)

            store_path = join(save_path, name_points + '.label')
            pred = pred + 1
            pred = self.remap_lut[pred].astype(np.uint32)
            pred.tofile(store_path)

    def get_split_list(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The split name should be one of
            'training', 'test', 'validation', or 'all'.
        """
        cfg = self.cfg
        dataset_path = cfg.dataset_path

        seq_list = []
        if split in ['train', 'training']:
            # seq_list = cfg.training_split
            # train_file_names = "data/WHU-Railway3D/urban_label-static/model-train.txt"
            train_file_names = cfg.train_file_names
            with open(train_file_names, 'r') as file:
                for line in file:
                    seq_list.append(line.strip())
        elif split in ['test', 'testing']:
            # seq_list = cfg.test_split
            # test_file_names = "data/WHU-Railway3D/urban_label-static/model-test.txt"
            test_file_names = cfg.test_file_names
            with open(test_file_names, 'r') as file:
                for line in file:
                    seq_list.append(line.strip())

        elif split in ['val', 'validation']:
            # seq_list = cfg.validation_split
            # val_file_names = "data/WHU-Railway3D/urban_label-static/model-test.txt"
            val_file_names = cfg.val_file_names
            with open(val_file_names, 'r') as file:
                for line in file:
                    seq_list.append(line.strip())

        elif split in ['all']:
            seq_list = cfg.all_split
        else:
            raise ValueError("Invalid split {}".format(split))

        # seq_list = ["L7-1-M01-001"]

        # 使用进度条，描述文字：正在读取文件路径
        file_list = []
        for seq_id in tqdm(seq_list, desc=f"正在读取 {split} 文件路径"):
            pc_path = join(dataset_path, "cache", seq_id.split(".")[0] + ".npy")
            file_list.append(pc_path)
        # list 转为 numpy
        file_list = np.array(file_list)

        # 临时截取前100个文件即可 (这里获取的是每个点云文件)
        print(">>>> debug :")
        file_list = file_list[0:1]
        print(file_list)

        return file_list

    def get_data_list(self, path_list):
        # 使用进度条，描述文字：正在读取数据到内存
        data_list = []
        new_path_list = []
        for path in tqdm(path_list, desc=f"读取数据到内存"):
            data = np.load(path, allow_pickle=True).item()  # data： feat, label, point, search_tree

            # 读取每个点云文件中的点云，然后开始采样
            num_points = 65536
            sample_time = int((data["point"].shape[0] * 0.8) // num_points)  # 整数
            sub_list, sub_index = self.pre_sample(data["point"], sample_time, num_points)

            temp_data = {}
            for i in range(sample_time):
                temp_data["point"] = sub_list[i]
                temp_data["label"] = data["label"][sub_index[i]]
                temp_data["feat"] = None if data["feat"] is None else data["feat"][sub_index[i]]
                temp_data["search_tree"] = None
                data_list.append(temp_data)
                new_path_list.append(path)

                # 临时存储数据，检查数据
                out_file = path.split("/")[-1].split(".")[0] + f"_{i}.las"
                save_dir = "/home/gisleung/dataset2/Leung-Rail/check/"
                create_new_las(os.path.join(save_dir, out_file), sub_list[i], temp_data["label"])


        # list 转 np
        if len(new_path_list) > 0:
            new_path_list = np.array(new_path_list)
        else:
            new_path_list = path_list

        return data_list, new_path_list

    # 预先采样
    def pre_sample(self, pc, num_sub=1, num_points=2048):
        pc_xoy = pc.copy()
        pc_xoy[:, 2] = 0
        search_tree = KDTree(pc_xoy)
        possibilities = np.random.rand(pc_xoy.shape[0]) * 1e-3

        sub_list = []
        idx_list = []

        for j in range(num_sub):
            n = 0
            while n < 1000:
                center_id = np.argmin(possibilities)
                if possibilities[center_id] >= 0.1:
                    print(f"最小的可能性：{possibilities[center_id]}")
                center_point = pc_xoy[center_id, :].reshape(1, -1)
                idxs = search_tree.query(center_point, k=num_points)[1][0]
                n = len(idxs)
                if n < 1000:
                    possibilities[center_id] += 0.001

            sub_list.append(pc[idxs])
            idx_list.append(idxs)

            possibilities[idxs] += 0.3
            dists = np.sum(np.square((pc_xoy[idxs] - center_point).astype(np.float32)), axis=1)
            max_dist = np.sqrt(np.max(dists))
            idxs3 = search_tree.query_radius(center_point, r=max_dist * 1.5)[0]
            possibilities[idxs3] += 0.1

        return sub_list, idx_list


class LeungRailSplit(BaseDatasetSplit):

    def __init__(self, dataset, split='training'):
        # 在父类中 调用 dataset 的 get_split_list 初始化数据列表
        super().__init__(dataset, split=split)
        log.info("Found {} pointclouds for {}".format(len(self.path_list),
                                                      split))
        # self.remap_lut_val = dataset.remap_lut_val

    def __len__(self):
        return len(self.path_list)
        # return len(self.data_list)

    def get_data(self, idx):
        # 如果有缓存的话，应该不会走到这里
        # time1 = time.time()
        # pc_path = self.path_list[idx]
        # data = np.load(pc_path, allow_pickle=True).item()
        # print(f"索引 {idx} - {pc_path.split('/')[-1]} 读取点云数据耗时：{time.time() - time1} s")

        # time2 = time.time()
        data2 = self.data_list[idx]
        # print(f"索引 {idx} - self.data_list[idx] 读取点云数据耗时：{time.time() - time2} s")
        return data2

    def get_attr(self, idx):
        pc_path = self.path_list[idx]
        dir, file = split(pc_path)
        _, seq = split(split(dir)[0])
        name = '{}#{}'.format(seq, file[:-4])

        pc_path = str(pc_path)
        attr = {'idx': idx, 'name': name, 'path': pc_path, 'split': self.split}
        return attr


# register 应该可以直接用名称获取
DATASET._register_module(LeungRail)