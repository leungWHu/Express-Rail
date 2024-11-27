import time

import numpy as np
import random

from ...utils import SAMPLER


class SemSegRandomSampler(object):
    """Random sampler for semantic segmentation datasets.
        翻译: 采样器
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.length = len(dataset)
        self.split = self.dataset.split

    def __len__(self):
        return self.length

    def initialize_with_dataloader(self, dataloader):
        self.length = len(dataloader)

    def get_cloud_sampler(self):
        """ 从dataset中取出一个点云（A）的索引 """
        def gen():
            # 常规的，随机打乱索引，然后挨个迭代
            ids = np.random.permutation(self.length)
            for i in ids:
                yield i

        return gen()

    @staticmethod
    def get_point_sampler():
        """ 从 点云（A） 中取出 num_points 个点，以及一个中心点 """
        def _random_centered_gen(**kwargs):
            # TODO 这部分耗时有问题
            # 从点云中随机采样一个点, 并以该点为中心, 采样num_points个点
            pc = kwargs.get('pc', None)
            num_points = kwargs.get('num_points', None)
            radius = kwargs.get('radius', None)
            search_tree = kwargs.get('search_tree', None)
            if pc is None or num_points is None or search_tree is None:
                raise KeyError("Please provide pc, num_points, and search_tree \
                    for point_sampler in SemSegRandomSampler")

            # 下面是旧的代码，但是这里的代码有问题，因为没有考虑到半径查询
            center_idx = np.random.choice(len(pc), 1)
            center_point = pc[center_idx, :].reshape(1, -1)
            if (pc.shape[0] < num_points):
                diff = num_points - pc.shape[0]
                idxs = np.array(range(pc.shape[0]))
                idxs = list(idxs) + list(random.choices(idxs, k=diff))
                idxs = np.asarray(idxs)
            else:
                idxs = search_tree.query(center_point, k=num_points)[1][0]

            # 下面是新的代码，参考 SemSegSpatiallyRegularSampler
            n = 0
            while n < 2:
                center_idx = np.random.choice(len(pc), 1)
                center_point = pc[center_idx, :].reshape(1, -1)

                if radius is not None:
                    idxs = search_tree.query_radius(center_point, r=radius)[0]
                elif num_points is not None:
                    if (pc.shape[0] < num_points):
                        diff = num_points - pc.shape[0]
                        idxs = np.array(range(pc.shape[0]))
                        idxs = list(idxs) + list(random.choices(idxs, k=diff))
                        idxs = np.asarray(idxs)
                    else:
                        idxs = search_tree.query(center_point,
                                                 k=num_points)[1][0]
                n = len(idxs)

            random.shuffle(idxs)
            pc = pc[idxs]

            # time001 = time.time()
            # print(f"【时间戳001】点数 {count_temp}：{time001} - 【debug】：_random_centered_gen 函数完成")

            return pc, idxs, center_point
        return _random_centered_gen


SAMPLER._register_module(SemSegRandomSampler)
