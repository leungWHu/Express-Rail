# -*- coding: utf-8 -*-
import argparse
import os
import time
import warnings

from My_utils.common_util import get_max_free_gpu

parser = argparse.ArgumentParser()
# 必填项
parser.add_argument('--data', type=str, default=None, help='Dataset name.')
parser.add_argument('--gpu', type=int, default=-1, help='GPU to use [default: -1].')
FLAGS = parser.parse_args()

# 设置os.environ['CUDA_VISIBLE_DEVICES']必须在导入torch之前，ml3d中存在torch
gpu_id = FLAGS.gpu
if gpu_id == -1:
    # 设置占用的GPU
    gpu_id = get_max_free_gpu()  # '0'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # Set GPU visible device
print(f'- - - * - * - * -Using GPU {gpu_id}')

import torch
import ml3d as _ml3d
from ml3d.datasets import ExpressRail, WhuRail, SNCFRail
from ml3d.torch import DGCNN, SemanticSegmentation

print("Available devices:", torch.cuda.device_count())

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    # 屏蔽所有警告
    warnings.filterwarnings("ignore")

    run_mode = "train"
    # run_mode = "test"
    log_path = "logs_dgcnn/1112-1523_ExpressRail"

    # ExpressRail, WhuRailway3D, SNCFRail
    datasetName = FLAGS.data if FLAGS.data is not None else "ExpressRail"
    print(f"数据集：{datasetName}, 运行模式：{run_mode}，请注意.... \n")
    time.sleep(3)

    cfg_file = f"dataset/{datasetName}/config_DGCNN.yaml" if run_mode == "train" else f"{log_path}/config.yaml"

    cfg = _ml3d.utils.Config.load_from_file(cfg_file)
    base_info = _ml3d.utils.Config.load_from_file(f"dataset/{datasetName}/base_info.yaml")
    cfg.dataset.dataset_path = cfg.dataset.dataset_path.replace("$down_sample$", str(cfg.dataset.down_sample))
    cfg.dataset.class_weights = base_info[f"count_test_{cfg.dataset.down_sample}"]

    # 创建数据集
    if datasetName == "ExpressRail":
        dataset = ExpressRail(cfg.dataset.pop('dataset_path', None), info=base_info, **cfg.dataset)
    elif datasetName == "WhuRailway3D":
        dataset = WhuRail(cfg.dataset.pop('dataset_path', None), info=base_info, **cfg.dataset)
    elif datasetName == "SNCFRail":
        dataset = SNCFRail(cfg.dataset.pop('dataset_path', None), info=base_info, **cfg.dataset)
    else:
        raise ValueError(f"未知数据集：{datasetName}")

    """ > > > >> >> >>> run_train <<< << << < < <"""
    if run_mode == "train":
        print(" 训练模式： ！！！ ")

        model = DGCNN(**cfg.model)  # 创建模型 无需.cuda()，先放在cpu，semantic_segmentation.py会切换到gpu
        pipeline = SemanticSegmentation(model, dataset=dataset, device="gpu", use_cache=True,
                                        info=base_info, **cfg.pipeline)
        pipeline.run_train()

    """ > > > >> >> >>> run_test <<< << << < < <"""
    if run_mode == "test":
        print(" 测试模式： ！！！ 请检查主路径配置文件中的测试参数！")

        test_ckpt_path = cfg.pipeline.get('test_ckpt_path', None)
        print(f"加载模型参数: {test_ckpt_path}")
        cfg.pipeline.test_result_folder = test_ckpt_path.split("/checkpoint/")[0]

        model = DGCNN(test_ckpt_path=test_ckpt_path, **cfg.model)
        pipeline = SemanticSegmentation(model, dataset=dataset, info=base_info, split="test", device="gpu",
                                        **cfg.pipeline)
        pipeline.run_test()

    print('end!')
