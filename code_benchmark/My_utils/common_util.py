import os
import shutil
import GPUtil

import numpy as np
import torch
import logging

import yaml
import matplotlib.pyplot as plt
import seaborn as sns

def readYamlFile(filepath):
    # 打开并读取YAML文件
    with open(filepath, "r") as file:
        data = yaml.safe_load(file)
    return data


def saveYamlFile(filepath, dict_data):
    # 将数据存储为YAML文件
    with open("example.yaml", "w") as file:
        yaml.dump(dict_data, file)


def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def check_makedirs(dir_name):
    """
        是否存在，若不存在，直接创建
    """
    if not os.path.exists(dir_name):
        print(f"文件夹：{dir_name}不存在，已自动创建")
        os.makedirs(dir_name)


def get_dataset_info(dataName, key=None):
    """
        获取数据集的元数据
    """
    cfg = readYamlFile("data/{}/visualize.yaml".format(dataName))
    cfg = cfg[dataName]
    if key is None:
        return cfg
    else:
        return cfg[key]


def copy_file_or_dir(source, destination, reName=None):
    """
    中国化的文件（夹）复制操作
    @param source: 支持 文件夹 和 文件 2种类型
    @param destination: 只支持文件夹（可自动创建），并作为父级目录存放 source
    @param reName: 是否需要重命名（当 source 时文件时， 注意填写后缀）
    @return:
    """
    if source.endswith('/'):
        source = source[:-1]
    if destination.endswith('/'):
        destination = destination[:-1]

    if not os.path.exists(source):  # 检查源文件或文件夹是否存在
        print(f"复制出错：文件（夹）{source} 不存在")
        return

    os.makedirs(destination, exist_ok=True)  # 创建目标文件夹，如果不存在则自动创建

    if os.path.isdir(source):  # 复制文件夹
        if reName is None:
            shutil.copytree(source, os.path.join(destination, os.path.basename(source)))
        else:
            shutil.copytree(source, os.path.join(destination, reName))
        print(f"文件夹 '{source}' 已经复制到 '{destination}'.")
    else:  # 复制文件
        if reName is None:
            shutil.copy(source, destination)
            print(f"文件 '{source}' 已经复制到 '{destination}'.")
        else:
            destination_file = os.path.join(destination, reName)
            # shutil.copy2(source, destination_file)  # shutil.copy2 可以保留元数据（时间戳等），shutil.copy值复制文件内容
            shutil.copy(source, destination_file)  # shutil.copy2 可以保留元数据（时间戳等），shutil.copy值复制文件内容
            print(f"文件 '{source}' 已经复制到 '{destination_file}'.")


def get_logger(file_name):
    log_instance = logging.getLogger('mylogger')
    formatter = logging.Formatter('%(asctime)s [%(levelname)s]:  %(message)s',
                                  '%Y-%m-%d %H:%M:%S')
    # 写入日志文件的内容
    handler = logging.FileHandler(file_name, encoding="utf-8", mode="a")
    handler.setFormatter(formatter)
    log_instance.addHandler(handler)
    # 在控制台输出同样的内容
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    log_instance.addHandler(console_handler)

    log_instance.setLevel(logging.DEBUG)
    return log_instance


def tranTime2HMS(secondsNum, formatStr=False):
    """
    将 秒 转为 时分秒
    Args:
        secondsNum: 秒数
        formatStr: 是否格式化
    """
    hours, remainder = divmod(int(secondsNum), 3600)
    minutes, seconds = divmod(remainder, 60)
    if formatStr:
        return '{:02d}:{:02d}:{:02d}'.format(int(hours), int(minutes), int(seconds))
    else:
        return hours, minutes, seconds


def writer_txt_fit(pc_array, out_filename):
    # 输出点云txt (变长)
    # out_file = "exp_dir/output/train_visualize.txt"
    with open(out_filename, "w") as file:
        for i in range(pc_array.shape[0]):
            # 将每个点云特征转换为字符串形式
            point = pc_array[i]
            point_str = " ".join(str(feature) for feature in point)
            # 写入到文件中
            file.write(point_str + "\n")


def get_gpu_free_memory_percent(device_id):
    gpus = GPUtil.getGPUs()
    gpu = gpus[device_id]
    total_memory = gpu.memoryTotal
    free_memory = gpu.memoryFree
    return free_memory * 100 // total_memory


def get_free_gpu_with_memory_threshold(threshold=50):
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        free_percent = get_gpu_free_memory_percent(i)
        if free_percent > threshold:
            return i
    return -1  # 没有符合条件的 GPU


def get_max_free_gpu():
    # 获取可用显存最大GPU
    gpus = GPUtil.getGPUs()
    free_memory_list = [gpu.memoryFree for gpu in gpus]
    max_free_idx = np.argmax(free_memory_list)
    # 获取内存(2位小数)
    max_memory = round(free_memory_list[max_free_idx] / 1024, 2)  # G
    print(f"GPU-{max_free_idx} 可用空间最大：{max_memory}G")
    return max_free_idx


def copy_folder(src, dst, ignore_list):
    """
    复制文件夹，忽略某些文件
    Args:
        src:
        dst:
        ignore_list:
    Returns:

    """
    def ignore_files(dirname, filenames):
        return [name for name in filenames if name in ignore_list]

    shutil.copytree(src, dst, ignore=ignore_files)


class AverageMeter(object):
    """Computes and stores the average and current value 计算并存储平均值和当前值"""

    def __init__(self):
        self.count = None
        self.sum = None
        self.avg = None
        self.val = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def visualize_cm(cm, save_path, class_labels):
    # 归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # # 保留两位小数
    # cm_normalized = np.round(cm_normalized, 2)
    # print(cm_normalized)

    # 可视化归一化混淆矩阵（纵轴和横轴调换，去除标题，横轴标签在上方）
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm_normalized, annot=True, fmt=".4f", cmap='Blues', xticklabels=class_labels,
                     yticklabels=class_labels)
    plt.xlabel('Pre Labels', fontsize=14, fontweight='bold')
    plt.ylabel('True Labels', fontsize=14, fontweight='bold')

    # 调整横轴标签位置
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    # 将纵轴标签设置为水平显示
    plt.yticks(rotation=0)

    # 调整横轴标题位置
    ax.xaxis.set_label_coords(0.5, 1.08)

    # 调整图的边距，向右移动主体
    plt.subplots_adjust(left=0.15, right=1.0)

    plt.title('')
    # 保存图像
    plt.savefig(save_path)


class MyDumper(yaml.Dumper):
    """ 在写入 yaml 文件时，将list数据写为行内模式 """
    def represent_sequence(self, tag, sequence, flow_style=False):
        return super().represent_sequence(tag, sequence, flow_style=True)


def calculate_acc(cm):
    # 计算每个类别的准确率
    class_accuracies = np.diag(cm) / cm.sum(axis=1)
    # 计算平均准确率
    mean_accuracy = np.mean(class_accuracies)

    return class_accuracies, mean_accuracy


def calculate_oa(cm):
    """
    计算总体OA
    """
    return np.sum(np.diag(cm)) / np.sum(cm)


if __name__ == '__main__':
    # 生成一个100～200之间的随机数，形状为11*11
    data = np.random.randint(100, 200, (11, 11))
    names = ['Rails', 'Bed', 'Masts', 'Support',
          'Overhead', 'Fences', 'Poles', 'Veget', 'Build', 'Ground', 'Others']

    sava_path = 'aaaa_exp_dir'
    # 如果不存在，则创建
    check_makedirs(sava_path)

    visualize_cm(data, f'{sava_path}/cm.png', names)