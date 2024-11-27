import logging
import os.path
import shutil
import time
from os.path import exists, join
from pathlib import Path
from datetime import datetime

import numpy as np
import yaml
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# pylint: disable-next=unused-import
from open3d.visualization.tensorboard_plugin import summary

from My_utils.common_util import copy_folder, tranTime2HMS, AverageMeter, visualize_cm, copy_file_or_dir, MyDumper
from .base_pipeline import BasePipeline
from ..dataloaders import get_sampler, RailDataloader, DefaultBatcher, ConcatBatcher
from ..utils import latest_torch_ckpt
from ..modules.losses import SemSegLoss, filter_valid_label
from ..modules.metrics import SemSegMetric
from ...utils import make_dir, PIPELINE, get_runid, code2md
from ...datasets import InferenceDummySplit

# LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
# DATE_FORMAT = '%Y%m%d %H:%M:%S'
# logging.basicConfig(format=LOGGING_FORMAT, datefmt=DATE_FORMAT)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class SemanticSegmentation(BasePipeline):
    """This class allows you to perform semantic segmentation for both training
    and inference using the Torch. This pipeline has multiple stages: Pre-
    processing, loading dataset, testing, and inference or training.

    **Example:**
        This example loads the Semantic Segmentation and performs a training
        using the SemanticKITTI dataset.

            import torch
            import torch.nn as nn

            from .base_pipeline import BasePipeline
            from torch.utils.tensorboard import SummaryWriter
            from ..dataloaders import get_sampler, RailDataloader, DefaultBatcher, ConcatBatcher

            Mydataset = RailDataloader(dataset=dataset.get_split('training')),
            MyModel = SemanticSegmentation(self,model,dataset=Mydataset, name='SemanticSegmentation',
            name='MySemanticSegmentation',
            batch_size=4,
            val_batch_size=4,
            test_batch_size=3,
            max_epoch=100,
            learning_rate=1e-2,
            lr_decays=0.95,
            save_ckpt_freq=20,
            adam_lr=1e-2,
            scheduler_gamma=0.95,
            momentum=0.98,
            main_log_dir='./logs/',
            device='gpu',
            split='train',
            train_sum_dir='train_log')

    **Args:**
            dataset: The 3D ML dataset class. You can use the base dataset, sample datasets , or a custom dataset.
            model: The model to be used for building the pipeline.
            name: The name of the current training.
            batch_size: The batch size to be used for training.
            val_batch_size: The batch size to be used for validation.
            test_batch_size: The batch size to be used for testing.
            max_epoch: The maximum size of the epoch to be used for training.
            leanring_rate: The hyperparameter that controls the weights during training. Also, known as step size.
            lr_decays: The learning rate decay for the training.
            save_ckpt_freq: The frequency in which the checkpoint should be saved.
            adam_lr: The leanring rate to be applied for Adam optimization.
            scheduler_gamma: The decaying factor associated with the scheduler.
            momentum: The momentum that accelerates the training rate schedule.
            main_log_dir: The directory where logs are stored.
            device: The device to be used for training.
            split: The dataset split to be used. In this example, we have used "train".
            train_sum_dir: The directory where the trainig summary is stored.

    **Returns:**
            class: The corresponding class.
    """

    def __init__(
            self,
            model,
            dataset=None,
            name='SemanticSegmentation',
            batch_size=4,
            val_batch_size=4,
            test_batch_size=3,
            max_epoch=100,  # maximum epoch during training
            start_epoch=1,
            learning_rate=1e-2,  # initial learning rate
            lr_decays=0.95,
            save_ckpt_freq=20,
            adam_lr=1e-2,
            scheduler_gamma=0.95,
            momentum=0.98,
            main_log_dir='./logs/',
            device='cuda',  # "gpu", "cuda" | "cpu"
            split='train',
            train_sum_dir=None,
            **kwargs):

        super().__init__(model=model,
                         dataset=dataset,
                         name=name,
                         batch_size=batch_size,
                         val_batch_size=val_batch_size,
                         test_batch_size=test_batch_size,
                         max_epoch=max_epoch,
                         start_epoch=start_epoch,
                         learning_rate=learning_rate,
                         lr_decays=lr_decays,
                         save_ckpt_freq=save_ckpt_freq,
                         adam_lr=adam_lr,
                         scheduler_gamma=scheduler_gamma,
                         momentum=momentum,
                         main_log_dir=main_log_dir,
                         device=device,
                         split=split,
                         train_sum_dir=train_sum_dir,
                         **kwargs)
        # if train_sum_dir is None:
        #     self.cfg.train_sum_dir = join(self.cfg.main_log_dir, 'train_log')
        self.best_iou_val = {'epoch': 0, 'value': 0}
        self.best_acc_val = {'epoch': 0, 'value': 0}

    def run_inference(self, data):
        """Run inference on given data. 对给定的数据运行推理。

        Args:
            data: A raw data.

        Returns:
            Returns the inference results.
        """
        cfg = self.cfg
        model = self.model
        device = self.device

        model.to(device)
        model.device = device
        model.eval()

        batcher = self.get_batcher(device)
        infer_dataset = InferenceDummySplit(data)
        self.dataset_split = infer_dataset
        infer_sampler = infer_dataset.sampler
        infer_split = RailDataloader(dataset=infer_dataset,
                                     preprocess=None,
                                     transform=model.transform,
                                     sampler=infer_sampler,
                                     use_cache=False)
        infer_loader = DataLoader(infer_split,
                                  batch_size=cfg.batch_size,
                                  sampler=get_sampler(infer_sampler),
                                  collate_fn=batcher.collate_fn)

        model.trans_point_sampler = infer_sampler.get_point_sampler()
        self.curr_cloud_id = -1
        self.test_probs = []
        self.ori_test_probs = []
        self.ori_test_labels = []

        with torch.no_grad():
            for unused_step, inputs in enumerate(infer_loader):
                """ 推理 """
                results = model(inputs['data'])
                self.update_tests(infer_sampler, inputs, results)

        inference_result = {
            'predict_labels': self.ori_test_labels.pop(),
            'predict_scores': self.ori_test_probs.pop()
        }

        metric = SemSegMetric()

        valid_scores, valid_labels = filter_valid_label(
            torch.tensor(inference_result['predict_scores']),
            torch.tensor(data['label']), model.cfg.num_classes,
            model.cfg.ignored_label_inds, device)

        metric.update(valid_scores, valid_labels)
        log.info(f"Accuracy : {metric.acc()}")
        log.info(f"IoU : {metric.iou()}")

        return inference_result

    def run_test(self):
        """Run the test using the data passed. 使用传递的数据运行测试。"""
        model = self.model
        dataset = self.dataset
        device = self.device
        cfg = self.cfg
        model.device = device
        model.to(device)
        model.eval()

        # 初始化 log
        timestamp = datetime.now().strftime('%m%d-%H%M')
        # self.dataset.cfg.test_result_folder = join(self.dataset.cfg.test_result_folder, f'testResult_{timestamp}')
        self.dataset.cfg.test_result_folder = join(self.cfg.test_result_folder, f'testResult_{timestamp}')
        make_dir(self.dataset.cfg.test_result_folder)

        self.init_log(timestamp)
        log.info("DEVICE : {}".format(device))

        test_dataset = dataset.get_split('test')
        test_sampler = test_dataset.sampler  # 测试时，=SemSegSpatiallyRegularSampler
        # 下面这个是Dataset的子类，不是loader
        test_split = RailDataloader(dataset=test_dataset,
                                    preprocess=None,
                                    transform=model.transform,
                                    sampler=test_sampler,
                                    use_cache=dataset.cfg.use_cache)
        batcher = self.get_batcher(device)
        test_loader = DataLoader(test_split,
                                 # batch_size=cfg.test_batch_size,
                                 batch_size=cfg.test_batch_size,
                                 sampler=get_sampler(test_sampler),
                                 collate_fn=batcher.collate_fn)

        self.dataset_split = test_dataset

        self.load_ckpt(model.cfg.test_ckpt_path, False)
        # 复制 checkpoint 到 self.dataset.cfg.test_result_folder
        copy_file_or_dir(model.cfg.test_ckpt_path, self.dataset.cfg.test_result_folder)

        model.trans_point_sampler = test_sampler.get_point_sampler()
        self.curr_cloud_id = -1
        self.test_probs = []
        self.ori_test_probs = []
        self.ori_test_labels = []

        record_summary = cfg.get('summary').get('record_for', [])
        log.info("Started testing")

        """
            一个终极方案：原始点124605,体素采样83891，随机采样送入网络45056
            测试时，是如何保证每个点都被预测到的？
                1. 原始点云体素采样至 83891 个点
                2. 随机采样45056个点（每组对同一个点云采样batch_size次），送入网络，得到预测结果
                3. 判断之前所有组的预测结果，是否覆盖了 83891 个点 ？ 
                    a. 如果否，反复执行2和3，直到覆盖了 83891 个点；
                    b. 如果是，则将 83891 映射到 124605 个点，计算准确率和IoU
                
        """
        self.metric_test_seq = SemSegMetric()  # 一个序列的测试结果
        self.metric_test_all = SemSegMetric("all")  # 全局的测试结果
        item_time = AverageMeter()
        end = time.time()
        cur_name_seq = None
        with torch.no_grad():
            print(" * * * * * 开始测试 * * * * * ")
            """ test_loader（gen_test） 控制 迭代行为，结合 update_tests 函数。实现全预测 """
            for unused_step, inputs in enumerate(test_loader):

                # randlanet.py 中的 forward 函数 会将输入数据转为cuda对象
                # for key in inputs['data']:
                #     # 将tensor数据转为cuda对象
                #     if type(inputs['data'][key]) is list:
                #         for i in range(len(inputs['data'][key])):
                #             inputs['data'][key][i] = inputs['data'][key][i].to(device)
                #     else:
                #         inputs['data'][key] = inputs['data'][key].to(device)

                if hasattr(inputs['data'], 'to'):
                    inputs['data'].to(device)
                """ 测试时 """
                results = model(inputs['data'])

                # 重点：怎么就从预测结果，扩展到原始点了呢？原始点124605,体素采样83891，随机采样送入网络45056
                self.update_tests(test_sampler, inputs, results)

                # 当一个场景的所有点都被预测到时
                if self.complete_infer:
                    inference_result = {
                        'predict_labels': self.ori_test_labels.pop(),
                        'predict_scores': self.ori_test_probs.pop()
                    }
                    attr = self.dataset_split.get_attr(test_sampler.cloud_id)
                    prepare_data = self.dataset_split.get_data(test_sampler.cloud_id)

                    # gt_labels = prepare_data['label']
                    gt_labels = prepare_data['ori_point_label'][:, -1]  # 转为int32
                    gt_labels = gt_labels.astype(np.int32)  # 转为int32

                    if (gt_labels > 0).any():
                        valid_scores, valid_labels = filter_valid_label(
                            torch.tensor(inference_result['predict_scores']).to(device),
                            torch.tensor(gt_labels).to(device),
                            model.cfg.num_classes,
                            model.cfg.ignored_label_inds,
                            device)

                        if cur_name_seq is not None and attr["name"].split("#")[0] != cur_name_seq:
                            # 计算上一个序列的准确率和IoU
                            self.complete_test_one(cur_name_seq)
                            # 开启新序列的计算
                            self.metric_test_seq.reset()

                        # 最后一个序列
                        if len(test_sampler.dataset) == self.curr_cloud_id + 1:
                            self.metric_test_seq.update(valid_scores, valid_labels)
                            self.complete_test_one(cur_name_seq)
                        else:
                            self.metric_test_seq.update(valid_scores, valid_labels)  # n*k, n

                        self.metric_test_all.update(valid_scores, valid_labels)
                        cur_name_seq = attr["name"].split("#")[0]

                    # 保存测试结果
                    dataset.save_test_result(inference_result, attr,
                                             prepare_data['ori_point_label'] if cfg.get('output_test', False) else None)

                    # Save only for the first batch
                    if 'test' in record_summary and 'test' not in self.summary:
                        self.summary['test'] = self.get_3d_summary(
                            results, inputs['data'], 0, save_gt=False)

                    """ 实时计算程序耗时 """
                    item_time.update(time.time() - end)
                    end = time.time()
                    remain_epoch = len(test_sampler.dataset) - (
                            self.curr_cloud_id + 1)  # self.curr_cloud_id, len(sampler.dataset)
                    remain_time = remain_epoch * item_time.avg  # 单位 秒s
                    use_time = tranTime2HMS(item_time.sum, True)
                    rem_time = tranTime2HMS(remain_time, True)
                    complete_time = time.strftime("%H:%M:%S", time.localtime(end + remain_time))
                    if (self.curr_cloud_id + 1) % 10 == 0:
                        print(
                            f" > 用 {use_time} ({round(item_time.avg, 0)}s/it)，需 {rem_time}，预 {complete_time} 完成 \n")

        log.info("")
        log.info("> > > Overall Testing Results < < <")
        log.info(f"      OA : {self.metric_test_all.OA()}")
        log.info(f"Accuracy : {self.metric_test_all.acc()}")
        log.info(f"     IoU : {self.metric_test_all.iou()}")
        log.info(f"       F1 : {self.metric_test_all.f1()}")

        # 因为测试阶段log时不在控制台输出，所以手动print
        result_str = (f"Testing : Overall Accuracy : {self.metric_test_all.OA()}, "
                      f"mAcc : {self.metric_test_all.acc()[-1]}, "
                      f"mIoU : {self.metric_test_all.iou()[-1]}, "
                      f" mf1 :{self.metric_test_all.f1()[-1]}")
        log.info(result_str)
        print(result_str)

        # 可视化总体的混淆矩阵（输出图片）
        cm_file = self.dataset.cfg.test_result_folder + "/confusion_matrix.jpg"
        visualize_cm(self.metric_test_all.confusion_matrix, cm_file, self.cfg.info.names)

        log.info("Finished testing")
        # log.info("测试结果存放在 : {}".format(self.dataset.cfg.test_result_folder))
        self.complete_work(self.dataset.cfg.test_result_folder)

    def complete_test_one(self, seq_name):
        """ 完成一个序列的测试时，打印结果，打印混淆矩阵
            （此专门针对铁路特殊场景）
        """
        log.info("")
        log.info(f"> > > ：{seq_name}  "
                 f"mAcc {self.metric_test_seq.acc()[-1]} "
                 f"mIoU {self.metric_test_seq.iou()[-1]}")
        log.info(f"Accuracy : {self.metric_test_seq.acc()}")
        log.info(f"     IoU : {self.metric_test_seq.iou()}")
        log.info("Confusion Matrix : ")

        # 打印每个序列的混淆矩阵
        data_info = self.cfg.get('info', None)
        confusion_matrix = self.metric_test_seq.confusion_matrix
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.round(cm_normalized, 4)
        if data_info is not None:
            line_0 = " " * 9
            for name in data_info["names"]:
                # 居中对齐
                line_0 += '{:^8s} '.format(name)
            log.info(line_0)
            for (data, idx) in zip(cm_normalized, range(len(cm_normalized))):
                line_item = '{:^8s} '.format(data_info["names"][idx])
                for item in data:
                    line_item += '{:^8.3f} '.format(item)
                log.info(line_item)

    def update_tests(self, sampler, inputs, results):
        """
        Update tests using sampler, inputs, and results. 使用采样器、输入和结果更新测试。
        @param sampler:
        @param inputs: 输入网络的分级数据
        @param results: 随机采样后的45056个点
        @return:
        """
        split = sampler.split
        end_threshold = 0.5
        cur_file_name = inputs["attr"]["name"][0].split('#')[-1]
        if self.curr_cloud_id != sampler.cloud_id:
            self.curr_cloud_id = sampler.cloud_id
            num_points = sampler.possibilities[sampler.cloud_id].shape[0]  # （下采样之后的点数）
            self.pbar = tqdm(total=num_points,
                             desc="进度 > ({}/{}) {} {} ".format(self.curr_cloud_id, len(sampler.dataset),
                                                          split, cur_file_name))
            self.pbar_update = 0  # 已经完成预测的点个数
            self.test_probs.append(np.zeros(shape=[num_points, self.model.cfg.num_classes], dtype=np.float16))
            self.complete_infer = False

        this_possiblility = sampler.possibilities[sampler.cloud_id]
        self.pbar.update(this_possiblility[this_possiblility > end_threshold].shape[0] - self.pbar_update)
        self.pbar_update = this_possiblility[this_possiblility > end_threshold].shape[0]  # 大于end_threshold的个数

        # 将部分预测结果更新到test_probs中
        self.test_probs[self.curr_cloud_id] = self.model.update_probs(
            inputs,
            results,
            self.test_probs[self.curr_cloud_id],
        )

        # 确保this_possiblility内的值都大于end_threshold：当体素采样的点都被预测后
        if (split in ['test'] and
                this_possiblility[this_possiblility > end_threshold].shape[0] == this_possiblility.shape[0]):

            # 将有限的预测点投影回原始全部点云
            # proj_inds = self.model.preprocess(
            #     self.dataset_split.get_data(self.curr_cloud_id),
            #     {'split': split}
            # ).get('proj_inds', None)
            prepare_data = self.dataset_split.get_data(self.curr_cloud_id)
            proj_inds = prepare_data.get('proj_inds', None)

            if proj_inds is None:
                proj_inds = np.arange(self.test_probs[self.curr_cloud_id].shape[0])

            test_labels = np.argmax(self.test_probs[self.curr_cloud_id][proj_inds], 1)

            # 全部点的预测分数
            self.ori_test_probs.append(self.test_probs[self.curr_cloud_id][proj_inds])
            # 全部点的预测标签
            self.ori_test_labels.append(test_labels)

            self.complete_infer = True

    def run_train(self):
        """  """
        # TODO torch.manual_seed的作用
        torch.manual_seed(self.rng.integers(np.iinfo(
            np.int32).max))  # Random reproducible seed for torch
        model = self.model
        device = self.device
        model.device = device
        dataset = self.dataset

        cfg = self.cfg
        model.to(device)

        # 日志格式
        self.init_log(None, True)
        # formatter = logging.Formatter('%(asctime)s [%(levelname)s]:  %(message)s',
        #                               '%Y-%m-%d %H:%M:%S')
        #
        # # 输出日志到文件
        # timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # log_file_path = join(cfg.logs_dir, 'log_train_' + timestamp + '.txt')
        # handler = logging.FileHandler(log_file_path)
        # handler.setFormatter(formatter)
        # log.addHandler(handler)
        # # 输出日志到控制台
        # log.addHandler(logging.StreamHandler())

        log.info("DEVICE : {}".format(device))

        Loss = SemSegLoss(self, model, dataset, device)
        self.metric_train = SemSegMetric()
        self.metric_val = SemSegMetric()

        self.batcher = self.get_batcher(device)

        train_dataset = dataset.get_split('train')
        train_sampler = train_dataset.sampler

        train_split = RailDataloader(dataset=train_dataset,
                                     preprocess=None,
                                     transform=model.transform,
                                     sampler=train_sampler,  # initialize_with_dataloader 初始化每个点云的可能性
                                     use_cache=dataset.cfg.use_cache,
                                     steps_per_epoch=dataset.cfg.get(
                                         'steps_per_epoch_train', None))

        # time001 = time.time()
        train_loader = DataLoader(
            train_split,
            batch_size=cfg.batch_size,
            sampler=get_sampler(train_sampler),  # yield出数据索引
            num_workers=cfg.get('num_workers', 2),
            pin_memory=cfg.get('pin_memory', True),
            collate_fn=self.batcher.collate_fn,  # 将数据转为tensor
            worker_init_fn=lambda x: np.random.seed(x + np.uint32(
                torch.utils.data.get_worker_info().seed))
        )  # numpy expects np.uint32, whereas torch returns np.uint64.

        # print(f" * * * * * 初始化训练数据集耗时：{time.time() - time001} s")

        valid_dataset = dataset.get_split('validation')
        valid_sampler = valid_dataset.sampler
        valid_split = RailDataloader(dataset=valid_dataset,
                                     preprocess=None,
                                     transform=model.transform,
                                     sampler=valid_sampler,
                                     use_cache=dataset.cfg.use_cache,
                                     steps_per_epoch=dataset.cfg.get(
                                         'steps_per_epoch_valid', None))

        valid_loader = DataLoader(
            valid_split,
            batch_size=cfg.val_batch_size,
            sampler=get_sampler(valid_sampler),  # yield出数据索引
            num_workers=cfg.get('num_workers', 2),
            pin_memory=cfg.get('pin_memory', True),
            collate_fn=self.batcher.collate_fn,
            worker_init_fn=lambda x: np.random.seed(x + np.uint32(
                torch.utils.data.get_worker_info().seed)))

        log.info(f"训练集共 {train_split.dataset.path_list.size} 个数据，"
                 f"验证集共 {valid_split.dataset.path_list.size} 个数据。")

        self.optimizer, self.scheduler = model.get_optimizer(cfg)

        # is_resume = model.cfg.get('is_resume', False)
        self.load_ckpt(model.cfg.ckpt_path)

        dataset_name = dataset.name if dataset is not None else ''
        if self.cfg.train_sum_dir is None:
            # 直接存放在备份目录下
            self.tensorboard_dir = self.cfg.logs_dir
        else:
            tensorboard_dir = join(
                self.cfg.train_sum_dir,
                model.__class__.__name__ + '_' + dataset_name + '_torch')
            runid = get_runid(tensorboard_dir)
            self.tensorboard_dir = join(self.cfg.train_sum_dir,
                                        runid + '_' + Path(tensorboard_dir).name)

        writer = SummaryWriter(self.tensorboard_dir)
        self.save_config(writer)
        log.info("Writing summary in {}. \n".format(self.tensorboard_dir))
        record_summary = cfg.get('summary').get('record_for', [])

        epoch_time = AverageMeter()
        end = time.time()
        use_time = '0:0:0'
        log.info(f"Started training at epoch: {cfg.start_epoch}")
        for epoch in range(cfg.start_epoch, cfg.max_epoch + 1):
            log.info(f'=== EPOCH {epoch:d}/{cfg.max_epoch:d} ===')
            model.train()
            self.metric_train.reset()
            self.metric_val.reset()
            self.losses = []
            # model.trans_point_sampler = train_sampler.get_point_sampler()
            model.trans_point_sampler = None if train_sampler is None else train_sampler.get_point_sampler()  # 球采样
            # 计算每个step的耗时
            # enumerate_time = time.time()
            # 共4541个文件，batch=4时，需要循环1136次才能完成一个epoch
            for step, inputs in enumerate(tqdm(train_loader, desc='training')):
                # for step, inputs in enumerate(train_loader):
                # print()
                # enumerate_use = time.time() - enumerate_time
                # model_time = time.time()

                # randlanet.py 中的 forward 函数 会将输入数据转为cuda对象
                # for key in inputs['data']:
                #     # 将tensor数据转为cuda对象
                #     if type(inputs['data'][key]) is list:
                #         for i in range(len(inputs['data'][key])):
                #             inputs['data'][key][i] = inputs['data'][key][i].to(device)
                #     else:
                #         inputs['data'][key] = inputs['data'][key].to(device)

                if hasattr(inputs['data'], 'to'):
                    inputs['data'].to(device)

                self.optimizer.zero_grad()

                """ 训练时 """
                # enumerate_time = time.time()
                results = model(inputs['data'])  # batch_size * num_points * num_classes
                # print(f" * * * 训练耗时：{time.time() - enumerate_time} s")

                loss, gt_labels, predict_scores = model.get_loss(
                    Loss, results, inputs, device)

                if predict_scores.size()[-1] == 0:
                    continue

                loss.backward()
                if model.cfg.get('grad_clip_norm', -1) > 0:
                    torch.nn.utils.clip_grad_value_(model.parameters(), model.cfg.grad_clip_norm)
                self.optimizer.step()

                self.metric_train.update(predict_scores, gt_labels)

                self.losses.append(loss.cpu().item())
                # Save only for the first pcd in batch
                if 'train' in record_summary and step == 0:
                    self.summary['train'] = self.get_3d_summary(
                        results, inputs['data'], epoch)
                # print(f" epoch={epoch} step={step}, emte_use={enumerate_use}, model_ues={time.time() - model_time}")
                #
                # enumerate_time = time.time()
                # print()

            self.scheduler.step()

            # --------------------- validation
            model.eval()
            self.valid_losses = []
            model.trans_point_sampler = valid_sampler.get_point_sampler()

            with torch.no_grad():
                # 共2271个 len(valid_loader)=2271  len(valid_split)=4541 val_batch_size=2
                for step, inputs in enumerate(tqdm(valid_loader, desc='validation')):

                    if hasattr(inputs['data'], 'to'):
                        inputs['data'].to(device)

                    """ 验证时 """
                    results = model(inputs['data'])
                    loss, gt_labels, predict_scores = model.get_loss(
                        Loss, results, inputs, device)

                    if predict_scores.size()[-1] == 0:
                        continue

                    self.metric_val.update(predict_scores, gt_labels)

                    self.valid_losses.append(loss.cpu().item())
                    # Save only for the first batch
                    if 'valid' in record_summary and step == 0:
                        self.summary['valid'] = self.get_3d_summary(
                            results, inputs['data'], epoch)

            self.save_logs(writer, epoch)

            if epoch % cfg.save_ckpt_freq == 0 or epoch == cfg.max_epoch:
                self.save_ckpt(epoch)

            """ 实时计算程序耗时 """
            epoch_time.update(time.time() - end)
            end = time.time()
            remain_epoch = cfg.max_epoch - epoch
            remain_time = remain_epoch * epoch_time.avg  # 单位 秒s
            use_time = tranTime2HMS(epoch_time.sum, True)
            rem_time = tranTime2HMS(remain_time, True)
            complete_time = time.strftime("%H:%M:%S", time.localtime(end + remain_time))
            print(f" > 用 {use_time}，需 {rem_time}，预 {complete_time} 完成")

        # self.delete_resume()

        log.info("")
        log.info(f"> > > Finished training, Time use {use_time}")
        log.info(f"Best validation mAcc: {self.best_acc_val['value']:.4f} at epoch {self.best_acc_val['epoch']}")
        log.info(f"Best validation mIoU: {self.best_iou_val['value']:.4f} at epoch {self.best_iou_val['epoch']}")

        # 重命名为 done
        self.complete_work(self.tensorboard_dir)

    def get_batcher(self, device, split='training'):
        """Get the batcher to be used based on the device and split. 根据设备和分割获取要使用的批处理器。"""
        batcher_name = getattr(self.model.cfg, 'batcher')

        if batcher_name == 'DefaultBatcher':
            batcher = DefaultBatcher()
        elif batcher_name == 'ConcatBatcher':
            batcher = ConcatBatcher(device, self.model.cfg.name)
        else:
            batcher = None
        return batcher

    def get_3d_summary(self, results, input_data, epoch, save_gt=True):
        """
        Create visualization for network inputs and outputs. 为网络输入和输出创建可视化。

        Args:
            results: Model output (see below).
            input_data: Model input (see below).
            epoch (int): step
            save_gt (bool): Save ground truth (for 'train' or 'valid' stages).

        RandLaNet:
            results (Tensor(B, N, C)): Prediction scores for all classes
            inputs_batch: Batch of pointclouds and labels as a Dict with keys:
                'xyz': First element is Tensor(B,N,3) points
                'labels': (B, N) (optional) labels

        SparseConvUNet:
            results (Tensor(SN, C)): Prediction scores for all classes. SN is
                total points in the batch.
            input_batch (Dict): Batch of pointclouds and labels. Keys should be:
                'point' [Tensor(SN,3), float]: Concatenated points.
                'batch_lengths' [Tensor(B,), int]: Number of points in each
                    point cloud of the batch.
                'label' [Tensor(SN,) (optional)]: Concatenated labels.

        Returns:
            [Dict] visualizations of inputs and outputs suitable to save as an
                Open3D for TensorBoard summary.
        """
        if not hasattr(self, "_first_step"):
            self._first_step = epoch
        label_to_names = self.dataset.get_label_to_names()
        cfg = self.cfg.get('summary')
        max_pts = cfg.get('max_pts')
        if max_pts is None:
            max_pts = np.iinfo(np.int32).max
        use_reference = cfg.get('use_reference', False)
        max_outputs = cfg.get('max_outputs', 1)
        input_pcd = []
        gt_labels = []
        predict_labels = []

        def to_sum_fmt(tensor, add_dims=(0, 0), dtype=torch.int32):
            sten = tensor.cpu().detach().type(dtype)
            new_shape = (1,) * add_dims[0] + sten.shape + (1,) * add_dims[1]
            return sten.reshape(new_shape)

        # Variable size point clouds
        if self.model.cfg['name'] in ('KPFCNN', 'KPConv'):
            batch_lengths = input_data.lengths[0].detach().numpy()
            row_splits = np.hstack(((0,), np.cumsum(batch_lengths)))
            max_outputs = min(max_outputs, len(row_splits) - 1)
            for k in range(max_outputs):
                blen_k = row_splits[k + 1] - row_splits[k]
                pcd_step = int(np.ceil(blen_k / min(max_pts, blen_k)))
                res_pcd = results[row_splits[k]:row_splits[k + 1]:pcd_step, :]
                predict_labels.append(
                    to_sum_fmt(torch.argmax(res_pcd, 1), (0, 1)))
                if self._first_step != epoch and use_reference:
                    continue
                pointcloud = input_data.points[0][
                             row_splits[k]:row_splits[k + 1]:pcd_step]
                input_pcd.append(
                    to_sum_fmt(pointcloud[:, :3], (0, 0), torch.float32))
                if torch.any(input_data.labels != 0):
                    gtl = input_data.labels[row_splits[k]:row_splits[k + 1]]
                    gt_labels.append(to_sum_fmt(gtl, (0, 1)))

        elif self.model.cfg['name'] in ('SparseConvUnet', 'PointTransformer'):
            if self.model.cfg['name'] == 'SparseConvUnet':
                row_splits = np.hstack(
                    ((0,), np.cumsum(input_data.batch_lengths)))
            else:
                row_splits = input_data.row_splits
            max_outputs = min(max_outputs, len(row_splits) - 1)
            for k in range(max_outputs):
                blen_k = row_splits[k + 1] - row_splits[k]
                pcd_step = int(np.ceil(blen_k / min(max_pts, blen_k)))
                res_pcd = results[row_splits[k]:row_splits[k + 1]:pcd_step, :]
                predict_labels.append(
                    to_sum_fmt(torch.argmax(res_pcd, 1), (0, 1)))
                if self._first_step != epoch and use_reference:
                    continue
                if self.model.cfg['name'] == 'SparseConvUnet':
                    pointcloud = input_data.point[k]
                else:
                    pointcloud = input_data.point[
                                 row_splits[k]:row_splits[k + 1]:pcd_step]
                input_pcd.append(
                    to_sum_fmt(pointcloud[:, :3], (0, 0), torch.float32))
                if getattr(input_data, 'label', None) is not None:
                    if self.model.cfg['name'] == 'SparseConvUnet':
                        gtl = input_data.label[k]
                    else:
                        gtl = input_data.label[
                              row_splits[k]:row_splits[k + 1]:pcd_step]
                    gt_labels.append(to_sum_fmt(gtl, (0, 1)))
        # Fixed size point clouds
        elif self.model.cfg['name'] in ('RandLANet', 'PVCNN'):  # Tuple input
            if self.model.cfg['name'] == 'RandLANet':
                pointcloud = input_data['xyz'][0]  # 0 => input to first layer
            elif self.model.cfg['name'] == 'PVCNN':
                pointcloud = input_data['point'].transpose(1, 2)
            pcd_step = int(
                np.ceil(pointcloud.shape[1] /
                        min(max_pts, pointcloud.shape[1])))
            predict_labels = to_sum_fmt(
                torch.argmax(results[:max_outputs, ::pcd_step, :], 2), (0, 1))
            if self._first_step == epoch or not use_reference:
                input_pcd = to_sum_fmt(pointcloud[:max_outputs, ::pcd_step, :3],
                                       (0, 0), torch.float32)
                if save_gt:
                    gtl = input_data.get('label',
                                         input_data.get('labels', None))
                    if gtl is None:
                        raise ValueError("input_data does not have label(s).")
                    gt_labels = to_sum_fmt(gtl[:max_outputs, ::pcd_step],
                                           (0, 1))
        else:
            raise NotImplementedError(
                "Saving 3D summary for the model "
                f"{self.model.cfg['name']} is not implemented.")

        def get_reference_or(data_tensor):
            if self._first_step == epoch or not use_reference:
                return data_tensor
            return self._first_step

        summary_dict = {
            'semantic_segmentation': {
                "vertex_positions": get_reference_or(input_pcd),
                "vertex_gt_labels": get_reference_or(gt_labels),
                "vertex_predict_labels": predict_labels,
                'label_to_names': label_to_names
            }
        }
        return summary_dict

    def save_logs(self, writer, epoch):
        """Save logs from the training and send results to TensorBoard. 从训练中保存日志并将结果发送到TensorBoard。"""
        train_accs = self.metric_train.acc()
        val_accs = self.metric_val.acc()

        train_ious = self.metric_train.iou()
        val_ious = self.metric_val.iou()

        loss_dict = {
            'Training loss': np.mean(self.losses),
            'Validation loss': np.mean(self.valid_losses)
        }
        acc_dicts = [{
            'Training accuracy': acc,
            'Validation accuracy': val_acc
        } for acc, val_acc in zip(train_accs, val_accs)]

        iou_dicts = [{
            'Training IoU': iou,
            'Validation IoU': val_iou
        } for iou, val_iou in zip(train_ious, val_ious)]

        for key, val in loss_dict.items():
            writer.add_scalar(key, val, epoch)
        for key, val in acc_dicts[-1].items():
            writer.add_scalar("{}/ Overall".format(key), val, epoch)
        for key, val in iou_dicts[-1].items():
            writer.add_scalar("{}/ Overall".format(key), val, epoch)

        log.info(f"Loss train: {loss_dict['Training loss']:.3f} "
                 f" eval: {loss_dict['Validation loss']:.3f}")
        log.info(f"Train OA: {acc_dicts[-1]['Training accuracy']:.3f}  mIoU: {iou_dicts[-1]['Training IoU']:.3f}")

        log.info(f" Eval OA: {acc_dicts[-1]['Validation accuracy']:.3f}  mIoU: {iou_dicts[-1]['Validation IoU']:.3f}")

        data_info = self.cfg.get('info', None)

        if data_info is not None:
            # 打印类名：Cfg.yaml_cfg中的class
            name_str = "\tname:"
            acc_str = "\t acc:"
            iou_str = "\t iou:"
            acc_str_eval = "\t acc:"
            iou_str_eval = "\t iou:"
            for i, name in enumerate(data_info["names"]):
                name_str += '{:^8s} '.format(name)
                acc_str += '{:^8.2f} '.format(100 * train_accs[i])
                iou_str += '{:^8.2f} '.format(100 * train_ious[i])

                acc_str_eval += '{:^8.2f} '.format(100 * val_accs[i])
                iou_str_eval += '{:^8.2f} '.format(100 * val_ious[i])

            # print(name_str, "\n", acc_str, "\n", iou_str)
            log.info(name_str)
            log.info(acc_str)
            log.info(iou_str)
            log.info("- - - eval - - - ")
            log.info(acc_str_eval)
            log.info(iou_str_eval)

        # 保存最好的 mIoU
        if val_ious[-1] >= self.best_iou_val['value']:
            self.best_iou_val['value'] = val_ious[-1]
            self.best_iou_val['epoch'] = epoch
            self.save_ckpt(epoch, 'iou')
            log.info(f" ! ! ! Best mIoU at Epoch {epoch} : {val_ious[-1]:.3f}")
        if val_accs[-1] >= self.best_acc_val['value']:
            self.best_acc_val['value'] = val_accs[-1]
            self.best_acc_val['epoch'] = epoch
            self.save_ckpt(epoch, 'oa')
            log.info(f" ! ! ! Best OA at Epoch {epoch} : {val_accs[-1]:.3f}")

        for stage in self.summary:
            for key, summary_dict in self.summary[stage].items():
                label_to_names = summary_dict.pop('label_to_names', None)
                writer.add_3d('/'.join((stage, key)),
                              summary_dict,
                              epoch,
                              max_outputs=0,
                              label_to_names=label_to_names)

    def load_ckpt(self, ckpt_path=None, is_resume=False):
        """
        Load a checkpoint. You must pass the checkpoint and indicate if you want to resume.
        加载检查点。必须传递检查点并指示是否要恢复。
        Args:
            ckpt_path: 检查点位置
            is_resume: 是否中断继续
        """
        train_ckpt_dir = join(self.cfg.logs_dir, 'checkpoint')
        make_dir(train_ckpt_dir)

        if ckpt_path is not None and not exists(ckpt_path):
            raise FileNotFoundError(f' ckpt {ckpt_path} not found')
        if ckpt_path is not None and exists(ckpt_path):
            log.info(f'Loading checkpoint {ckpt_path}')
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt and hasattr(self, 'optimizer'):
                log.info(f'Loading checkpoint optimizer_state_dict')
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt and hasattr(self, 'scheduler'):
                log.info(f'Loading checkpoint scheduler_state_dict')
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.best_iou_val = ckpt['val']['iou']['value']
            # TODO 思考：如果是恢复模式，理论上除了恢复模型参数，是不是也要恢复epoch？
            #           但是，即使恢复了epoch，后续训练时采样的随机性是否具有可复现性
            self.cfg.start_epoch = ckpt['epoch'] + 1

            log.info(f' !!! Loaded checkpoint {ckpt_path} at epoch {ckpt["epoch"]}')
        else:
            log.info(' !!! Initializing from scratch.【从头开始初始化】')

    def save_ckpt(self, epoch, best_type=None):
        """Save a checkpoint at the passed epoch.   在传递的时代保存检查点。"""
        if best_type is None:
            ckpt_name = f'ckpt_epoch_{epoch}.pth'
            self.delete_resume()
        else:
            ckpt_name = f"ckpt_best_{best_type}.pth"
        path_ckpt = join(self.cfg.logs_dir, 'checkpoint')
        make_dir(path_ckpt)
        torch.save(
            dict(epoch=epoch,
                 val={'oa': self.best_acc_val, 'iou': self.best_iou_val},
                 model_state_dict=self.model.state_dict(),
                 optimizer_state_dict=self.optimizer.state_dict(),
                 scheduler_state_dict=self.scheduler.state_dict()),
            join(path_ckpt, ckpt_name))

        log.info(f'Epoch {epoch}: save {ckpt_name} to {path_ckpt:s}')

    def delete_resume(self):
        """Delete the resume file when training is complete. 训练完成后删除恢复文件。"""
        path_ckpt = join(self.cfg.logs_dir, 'checkpoint')
        # 删除文件名含有 resume 的pth文件
        for file in os.listdir(path_ckpt):
            if "resume" in file:
                os.remove(join(path_ckpt, file))

    def save_config(self, writer):
        """Save experiment configuration with tensorboard summary.  使用tensorboard摘要保存实验配置。"""
        if hasattr(self, 'cfg_tb'):
            writer.add_text("Description/Open3D-ML", self.cfg_tb['readme'], 0)
            writer.add_text("Description/Command line", self.cfg_tb['cmd_line'],
                            0)
            writer.add_text('Configuration/Dataset',
                            code2md(self.cfg_tb['dataset'], language='json'), 0)
            writer.add_text('Configuration/Model',
                            code2md(self.cfg_tb['model'], language='json'), 0)
            writer.add_text('Configuration/Pipeline',
                            code2md(self.cfg_tb['pipeline'], language='json'),
                            0)

    def copy_code(self):
        if self.cfg.split == 'train':
            # 获取当前程序运行主目录（工作目录）
            work_dir = os.getcwd()
            to_path = join(work_dir, self.cfg.logs_dir, "checkCode")
            # 如果to_path存在 则删除
            if os.path.exists(to_path):
                shutil.rmtree(to_path)
            copy_folder(work_dir, to_path,
                        ['data', 'docs',
                         "logs", "logs_kpc", 'train_log', 'logs_pointnet2', 'logs_dgcnn',
                         '.idea', '.gitignore', 'README.md', 'model_zoo.md'])
            # 单独再copy一下yaml文件
            yaml_files = join(work_dir, "dataset", self.dataset.name, f"config_{self.model.cfg.name}.yaml")
            copy_file_or_dir(yaml_files, join(work_dir, self.cfg.logs_dir), "config.yaml")

    def complete_work(self, work_dir):
        os.rename(work_dir, work_dir + '_done')
        print(f"{self.cfg.split}结果存放在: {work_dir + '_done'}")

        """ 修改目录中config.yaml文件中的checkpoint路径，便于直接开始测试 """
        if self.cfg.split == 'train':
            yaml_file = join(work_dir + '_done',
                             "checkCode", "dataset", self.dataset.name,
                             f"config_{self.model.cfg.name}.yaml")
            # 读取 YAML 文件
            with open(yaml_file, 'r') as file:
                config = yaml.safe_load(file)

            # 更新需要更改的值
            config['pipeline']['test_ckpt_path'] = "../checkpoint/ckpt_best_iou.pth"
            # 将路径变为绝对路径
            config['dataset']['dataset_path'] = f"{os.getcwd()}/{self.dataset.cfg.dataset_path}"
            config['dataset']['train_file_names'] = f"{os.getcwd()}/{self.dataset.cfg.train_file_names}"
            config['dataset']['test_file_names'] = f"{os.getcwd()}/{self.dataset.cfg.test_file_names}"
            config['dataset']['val_file_names'] = f"{os.getcwd()}/{self.dataset.cfg.val_file_names}"
            # config['dataset']['cache_dir'] = f"{os.getcwd()}/{self.dataset.cfg.cache_dir}"

            # 写入更新后的内容到 YAML 文件
            with open(yaml_file, 'w') as file:
                # yaml.dump(config, file,  default_flow_style=True, allow_unicode=True)
                yaml.dump(config, file, Dumper=MyDumper, allow_unicode=True)

    def init_log(self, timestamp=None, StreamHandler=False):
        # 日志格式
        formatter = logging.Formatter('%(asctime)s [%(levelname)s]:  %(message)s',
                                      '%Y-%m-%d %H:%M:%S')

        # 输出日志到文件
        if timestamp is None:
            timestamp = datetime.now().strftime('%m%d-%H%M')  # ('%Y-%m-%d_%H-%M-%S')

        if self.cfg.split == 'train':
            log_file_path = join(self.cfg.logs_dir, f'log_{self.cfg.split}_{timestamp}.txt')
        elif self.cfg.split == 'test':
            log_file_path = join(self.dataset.cfg.test_result_folder, f'log_{self.cfg.split}_{timestamp}.txt')

        handler = logging.FileHandler(log_file_path)
        handler.setFormatter(formatter)
        log.addHandler(handler)
        if StreamHandler:
            # 输出日志到控制台
            log.addHandler(logging.StreamHandler())

        log.info("日志存放在 : {}".format(log_file_path))
        return timestamp


PIPELINE._register_module(SemanticSegmentation, "torch")
