import os
from datetime import datetime

import numpy as np
import yaml
import torch
from abc import ABC, abstractmethod

from os.path import join, exists, dirname, abspath

from My_utils.common_util import get_free_gpu_with_memory_threshold, get_max_free_gpu
# use relative import for being compatible with Open3d main repo
from ...utils import Config, make_dir


class BasePipeline(ABC):
    """Base pipeline class."""

    def __init__(self,
                 model,
                 dataset=None,
                 device='cuda',
                 distributed=False,
                 **kwargs):
        """Initialize.

        Args:
            model: A network model.
            dataset: A dataset, or None for inference model.
            device: 'cuda' or 'cpu'.
            distributed: Whether to use multiple gpus.
            kwargs:

        Returns:
            class: The corresponding class.
        """
        self.cfg = Config(kwargs)

        if kwargs['name'] is None:
            raise KeyError("Please give a name to the pipeline")
        self.name = self.cfg.name

        self.model = model
        self.dataset = dataset
        self.rng = np.random.default_rng(kwargs.get('seed', None))

        self.distributed = distributed
        if self.distributed and self.name == "SemanticSegmentation":
            raise NotImplementedError(
                "Distributed training not implemented for SemanticSegmentation!"
            )

        self.rank = kwargs.get('rank', 0)

        dataset_name = dataset.name if dataset is not None else ''
        timestamp = datetime.now().strftime('%m%d-%H%M')  # '%Y%m%d-%H%M%S'
        if self.cfg.split == 'train':
            if self.cfg.exp is not None:
                self.cfg.logs_dir = join(
                    self.cfg.main_log_dir,
                    self.cfg.exp,
                    # f"{timestamp}_{model.__class__.__name__}_{dataset_name}")
                    f"{timestamp}_{self.dataset.cfg.name}")
            else:
                self.cfg.logs_dir = join(
                    self.cfg.main_log_dir,
                    # f"{timestamp}_{model.__class__.__name__}_{dataset_name}")
                    f"{timestamp}_{self.dataset.cfg.name}")
            # f"{timestamp}_{model.__class__.__name__}_{dataset_name}_torch")
        elif self.cfg.split == 'test':
            test_ckpt_path = self.model.cfg.test_ckpt_path
            # dirname 父级目录
            self.cfg.logs_dir = dirname(dirname(test_ckpt_path))

        if self.rank == 0:
            make_dir(self.cfg.main_log_dir)
            make_dir(self.cfg.logs_dir)

        print(f" >>> >>> 保存路径：{self.cfg.logs_dir} <<< <<<")

        if device == 'cpu' or not torch.cuda.is_available():
            if distributed:
                raise NotImplementedError(
                    "Distributed training for CPU is not supported yet.")
            self.device = torch.device('cpu')
        else:
            if distributed:
                self.device = torch.device(device)
                print(f"Rank : {self.rank} using device : {self.device}")
                torch.cuda.set_device(self.device)
            else:
                # gpu_id = get_max_free_gpu()
                gpu_id = 0  # 在程序开始时，已经指定了使用一个GPU
                if gpu_id != -1:
                    self.device = torch.device(f'cuda:{gpu_id}')
                    # print(f" >>> >>> 选择的GPU：{gpu_id} <<< <<<")
                else:
                    # 直接抛出异常
                    raise Exception(" ！ ！ ！没有可用的GPU ！ ！ ！")

        self.summary = {}
        self.cfg.setdefault('summary', {})

        # 拷贝代码
        self.copy_code()

    @abstractmethod
    def run_inference(self, data):
        """Run inference on a given data.

        Args:
            data: A raw data.

        Returns:
            Returns the inference results.
        """
        return

    @abstractmethod
    def run_test(self):
        """Run testing on test sets."""
        return

    @abstractmethod
    def run_train(self):
        """Run training on train sets."""
        return

    @abstractmethod
    def copy_code(self):
        """复制代码."""
        return
