# 对ExpressRail数据集进行基准方法测试

## 介绍
* 在进行基准方法的测试时，我们借助 [Open3D-ML](https://github.com/isl-org/Open3D-ML) 库，将流行的基准方法集成到一个代码工程
* 这样做的好处是，所有方法使用了相同的数据预处理、输入策略、超参数设置等，保证了公平对比
* 请注意：代码还没经过严格的整理，我们将在近期重新上传可读性更强的代码。欢迎您关注！

## 1 安装运行环境
我们参考 Open3D-ML 的使用说明，最终的运行环境为：
* Pytorch=2.0.0
* Pytorch-cuda=11.7
* Python=3.10
* Open3d=0.18.0

创建Python虚拟环境是，我们遵循以下步骤：
```bash
# 创建虚拟
conda create -n py310_open3d-ml python=3.10
conda activate py310_open3d-ml

# 按需安装代码运行时的依赖，您可以在调试时，根据程序提示按需安装。
conda install ...

# 测试open3d安装成功
pip install open3d
python -c "import open3d.ml.torch as ml3d"
```


## 2 铁路点云前期预处理
我们在使用数据集中的数据正式训练网络之前，对每个场景进行了分块处理。  
分块目标是希望控制每个分块内的点云数量约为100000，而不是暴力的直接按照统一尺寸裁剪。  
预处理代码是 [prepare_ExpressRail.py](code_benchmark/dataset/ExpressRail/prepare/prepare_ExpressRail.py)。
预处理逻辑遵循以下步骤：
1. 逐文件读取每个铁路点云数据
2. 根据期望的下采样目标，如 grid_size=0.05m，对点云进行一次抽稀
3. 接着将场景点云旋转至水平方向。这是一个自动过程。
4. 根据期望的分段长度和点数量，先进行轨道方向按长度分块，然后进行垂直方向按数量分块。
5. 保存每个分块的数据，作为输入网络的基本单位。


## 3 配置文件

对于在ExpressRail数据集上的基准测试，其配置文件在`code_benchmark/dataset/ExpressRai`目录下，
包括用于不同方法的`*.yaml`配置文件。

## 4 运行代码
1. 训练。根据实际情况修改`config_randla.yaml`中的参数
```bash
python train_Fit_PointNet2.py
```

2. 测试。只需要修改`train_Fit_{Model}.py`中的 run_mode='test'后，运行 `python train_Fit_PointNet2.py` ，即是测试模式。

