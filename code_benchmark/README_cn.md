# 基准方法

[English Version](README.md)

## 介绍
* 在测试基准方法时，我们借助 [Open3D-ML](https://github.com/isl-org/Open3D-ML) 库，将常见的基准方法整合到一个统一的代码框架中。
* 这样做的好处是，所有方法都使用相同的数据预处理、输入策略和超参数设置，从而保证了公平对比。
* **请注意：**  代码尚未经过完全整理，我们将在近期上传更加规范和易读的版本。敬请关注！

## 1 安装运行环境
我们参考了 Open3D-ML 的使用说明，最终配置的运行环境如下：
* Pytorch = 2.0.0
* Pytorch-cuda = 11.7
* Python = 3.10
* Open3d = 0.18.0

创建 Python 虚拟环境时，请遵循以下步骤：
```bash
# 创建虚拟环境
conda create -n py310_open3d-ml python=3.10
conda activate py310_open3d-ml

# 根据需要安装代码运行时的依赖，您可以根据程序提示安装额外的依赖。
conda install ...

# 测试 Open3D 是否安装成功
pip install open3d
python -c "import open3d.ml.torch as ml3d"

```

## 2 铁路点云数据预处理
在使用数据集中的数据进行正式训练之前，我们对每个场景进行了分块处理。  
分块的目标是控制每个分块内的点云数量大约为 100,000，而不是简单地通过统一尺寸裁剪进行分割。  
预处理的代码位于 [prepare_ExpressRail.py](code_benchmark/dataset/ExpressRail/prepare/prepare_ExpressRail.py)，其处理逻辑包括以下步骤：
1. 逐文件读取每个铁路点云数据。
2. 根据期望的下采样目标（例如 grid_size=0.05m），对点云数据进行一次抽稀处理。
3. 将场景点云旋转至水平位置，这一过程是自动完成的。
4. 根据期望的分块长度和点数量，先按轨道方向分块，再按垂直方向按数量分块。
5. 保存每个分块的数据，作为输入网络的基本单位。

## 3 配置文件

针对 ExpressRail 数据集的基准测试，配置文件位于 `code_benchmark/dataset/ExpressRail` 目录下，  
包括适用于不同方法的 `*.yaml` 配置文件。

## 4 运行代码
1. **训练**：根据实际情况调整 `config_{model}.yaml` 配置文件中的参数，然后运行以下命令：
```bash
python train_Fit_PointNet2.py --data=ExpressRail
```
2. **测试**：  
修改 `train_Fit_{model}.py` 中的 `run_mode='test'`， 改为测试模式，  
修改 `config_{model}.yaml` 中的 `test_ckpt_path: path/to/your/checkpoint/*.pth`,  
然后运行：
```bash
python train_Fit_PointNet2.py --data=ExpressRail
```
