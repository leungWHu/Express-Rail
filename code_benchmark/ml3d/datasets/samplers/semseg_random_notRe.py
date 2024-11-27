import time

import numpy as np
import random

from ...utils import SAMPLER


class SemSegRandomNotReSampler(object):
    """
        与 SemSegRandomSampler 的区别在于：
            不需要get_point_sampler重新采样;
            会在构建数据集时，直接采样
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
            ids = np.random.permutation(self.length)
            for i in ids:
                yield i

        return gen()

    @staticmethod
    def get_point_sampler():
        def _random_centered_gen(**kwargs):
            """
            **kwargs
                feat: None
                label: n的行向量
                num_points: 采样点数
                pc: N*3的点云
                search_tree: pc的KDTree

            return:
                    pc: 采样后的点云 num_points*3
                    idxs: 采样后的点云的索引 num_points的行向量
                    center_point: 采样点的中心点 1*3
            """
            pc = kwargs.get('pc', None)
            num_points = kwargs.get('num_points', None)
            if pc is None or num_points is None:
                raise KeyError("Please provide pc, num_points, and search_tree \
                    for point_sampler in SemSegRandomSampler")
            # 直接生成顺序的索引
            idxs = np.arange(pc.shape[0])
            center_point = None  #
            return pc, idxs, center_point

        return _random_centered_gen


SAMPLER._register_module(SemSegRandomNotReSampler)
