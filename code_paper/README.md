# 我们贡献的方法
> 请注意： 目前我们的代码还在整理中，我们将尽快完成上传。

## 介绍
我们分析了当前主流弱监督方法在铁路场景下的挑战，具体包括以下几个方面：
1. **标签形式**：稀疏点标签无法提供代表性训练样本，子云级标签无法提供有效监督。
2. **分割方法**：主流方法倾向于构建复杂的网络结构，但未充分考虑铁路场景的特点。
3. **标签数量**：当前主流方法并未显著减少样本标注的工作量。
4. **分割结果**：当前主流方法在铁路重要设施的分割性能有限。

我们提出了基于主动学习的弱监督方法，具有以下优势：
- 仅需极少的标签点。
- 主动挑选代表性样本。
- 降低类别分布不均的影响。
- 显著提升铁路设施的分割性能。

## 1 安装运行环境

代码的运行环境为：
- **Pytorch**: 1.11.0
- **Pytorch-cuda**: 11.3
- **Python**: 3.7

### 安装步骤：
```bash
# 创建虚拟环境
conda create -n py37_torch1_11_cu11_3 python=3.7
conda activate py37_torch1_11_cu11_3

# 安装 Pytorch
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

# 编译扩展
cd cpp_wrappers
sh compile_wrappers.sh

# 安装其他依赖
# 根据程序提示，按需安装运行时依赖
conda install ...
```


## 2 配置文件

对于 **ExpressRail** 数据集，其配置文件位于 `code_paper/dataset/ExpressRail` 目录下的 `config_ExpressRail.yaml` 文件中。该文件包含以下内容：
- 数据集的基本信息
- 主动学习过程的设置
- 网络训练的超参数等

## 3 运行代码

### 1. 训练
使用以下命令启动训练：
```bash
python train_Express.py
```

### 2. 测试
使用以下命令进行测试：
```bash
python test_models.py --snap=path_to_train_log
```

