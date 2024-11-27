import numpy as np
from tqdm import tqdm
import random

from ...utils import SAMPLER


class SemSegSpatiallyRegularSampler(object):
    """Spatially regularSampler sampler for semantic segmentation datasets.
        翻译：用于语义分割数据集的空间正则采样器。
        当测试数据时使用
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.length = len(dataset)
        self.split = self.dataset.split

    def __len__(self):
        return self.length

    def initialize_with_dataloader(self, dataloader):
        """
            但初始化Dataloader时，会调用该方法：初始化每个点云的可能性
        """
        self.possibilities = []  # 用于存储每个点云的可能性（下采样后的点云）
        self.min_possibilities = []  # 用于存储每个点云的最小可能性 （下采样后的点云）

        self.length = len(dataloader)
        dataset = self.dataset

        for index in range(len(dataset)):
            attr = dataset.get_attr(index)
            if dataloader.cache_convert:
                data = dataloader.cache_convert(attr['name'])
            elif dataloader.preprocess:
                data = dataloader.preprocess(dataset.get_data(index), attr)  # 下采样、等
            else:
                data = dataset.get_data(index)

            pc = data['point']
            self.possibilities += [np.random.rand(pc.shape[0]) * 1e-3]  # 给列表添加元素：随机生成的数
            self.min_possibilities += [float(np.min(self.possibilities[-1]))]  # 给列表添加元素：随机生成的数的最小值

    def get_cloud_sampler(self):

        def gen_train():
            for i in range(self.length):
                self.cloud_id = int(np.argmin(self.min_possibilities))
                yield self.cloud_id

        def gen_test():
            """
            当使用迭代器时，控制当前要迭代的元素（如果不满足要求，则一直迭代当前数据）
                * gen_test返回一个sample的迭代器。只要 yield语句还会被执行，那么当前的迭代器就不会结束。
                * 当yield语句不再被执行，那么迭代器就会结束。
                * 可以看到：
                    1 当min_possibilities[cloud_id] < 0.5时，即当前场景的点云还未被完全覆盖时，curr_could_id不更新
                    2 一直 yield 未被更新的curr_could_id，那么当前场景就会一直被迭代
                    3 当min_possibilities[cloud_id] > 0.5时，即当前场景的点云已经被完全覆盖时，curr_could_id更新
                    4 重复1-3，直到 yield 语句不再被执行（curr_could_id = self.length），那么迭代器就会结束。
            """
            curr_could_id = 0
            # loop
            while curr_could_id < self.length:
                if self.min_possibilities[curr_could_id] > 0.5:
                    curr_could_id = curr_could_id + 1
                    continue

                self.cloud_id = curr_could_id

                yield self.cloud_id  # 生成器

        if self.split in ['train', 'validation', 'valid', 'training']:
            gen = gen_train
        else:
            gen = gen_test
        return gen()

    def get_point_sampler(self):
        """ 即 网络结构中的 self.trans_point_sampler() 对应的方法 """
        def _random_centered_gen(patchwise=True, **kwargs):
            if not patchwise:
                self.possibilities[self.cloud_id][:] = 1.
                self.min_possibilities[self.cloud_id] = 1.
                return
            pc = kwargs.get('pc', None)
            num_points = kwargs.get('num_points', None)
            radius = kwargs.get('radius', None)
            search_tree = kwargs.get('search_tree', None)
            if pc is None or num_points is None or (search_tree is None and
                                                    radius is None):
                raise KeyError(
                    "Please provide pc, num_points, and (search_tree or radius) \
                    for point_sampler in SemSegSpatiallyRegularSampler")

            cloud_id = self.cloud_id
            n = 0
            while n < 2:
                """ 抽取一个未预测点作为中心点 """
                # ? 这么说的话，batch_size=1比较好？经验证，多batch_size正常可行！
                center_id = np.argmin(self.possibilities[cloud_id])
                center_point = pc[center_id, :].reshape(1, -1)

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
                if n < 2:
                    # 说明上一个中心点选的不好（选到了偏远的噪点），重新随机选择一个中心点
                    self.possibilities[cloud_id][center_id] += 0.001

            random.shuffle(idxs)
            pc = pc[idxs]

            """ 为了更高效的覆盖所有点 """
            dists = np.sum(np.square((pc - center_point).astype(np.float32)),
                           axis=1)  # 计算点云中每个点到中心点的距离
            delta = np.square(1 - dists / np.max(dists))  # 值越大，说明离中心点越近
            self.possibilities[cloud_id][idxs] += delta
            # 上面的理解：一次采样之后，通过上面的操作，距中心点距离比较远（>0.5*max_dis）的，仍然有机会被选中作为下一次的中心点

            new_min = float(np.min(self.possibilities[cloud_id]))
            self.min_possibilities[cloud_id] = new_min

            return pc, idxs, center_point

        return _random_centered_gen


SAMPLER._register_module(SemSegSpatiallyRegularSampler)
