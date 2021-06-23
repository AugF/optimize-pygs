## 简要介绍

optimize_pygs是基于PyTorch Geometric(PyG 1.5.0)的工作。
针对基于采样的图神经网络训练与推理中存在的两大性能问题：
1. 采样流程额外开销大、评估步骤耗时较长
2. 不同批次间的内存波动大，某些极大值可能导致内存溢出风险。

针对以上两大性能问题，提出了两种流程优化方法:
1. 基于流水线并行的图神经网络训练与推理流程优化方法(neuroc_pygs/sec4_time)
2. 面向内存受限环境的图神经网络训练与推理流程优化方法(neuroc_pygs/sec5_memory, neuroc_pygs/sec6_cutting)

## 目录架构

```
neuroc_pygs       // 核心代码
    models              // 模型文件目录
    samplers            // 采样相关目录
        cuda_prefetcher.py   // !!! 采样流程的流水线优化核心代码
        ...
    sec4_time           // 基于流水线并行的图神经网络训练与推理流程优化方法
        epoch_opt_full_eval.py
        epoch_opt_full_train.py
        epoch_opt_full_eval.py
        epoch_opt_full_train.py
        opt_experiment.py   // !!! 训练与评估步骤的流水线优化    
        ...
    sec5_memory         // 面向内存受限环境的图神经网络训练与推理流程优化方法, 以训练阶段举例
    sec6_cutting        // 面向内存受限环境的图神经网络训练与推理流程优化方法, 以推理阶段举例
    utils               // 一些辅助文件
    __init__.py     
    base_experiment.py  // 基本运行文件
    configs.py          // 配置
    options.py          // 辅助文件
README.md         // 代码说明
requirements.txt  // 依赖库
setup.py          // python库安装文件
```

## 运行环境

- 硬件环境
    - 2 × NVIDIA Tesla T4 GPU( 16GB)
    - CentOS 7 server, 40 cores, 90GB
- 软件环境：
    - Python3.7.7
    - PyTorch1.5.0
    - CUDA10.1
    - PyTorchGeometric1.5.0
    - ogb1.2.3

## 安装说明

这里统一采用pip安装方式：
1. 安装anaconda（[官方文档](https://docs.anaconda.com/anaconda/install/index.html)），创建新的环境`conda create -n optimize-pygs python==3.7.7`，并激活`conda activate optimize-pygs`
2. 安装`PyTorch1.5.0`, [官方文档](https://pytorch.org/), 执行命令`pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html`
3. 安装`PyTorchGeomtric1.5.0`, [官方文档](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), [PyG1.5.0+cu101](https://pytorch-geometric.com/whl/torch-1.5.0+cu101.html)。
    ```
    pip install tools/torch_cluster-1.5.4-cp37-cp37m-linux_x86_64.whl
    pip install tools/torch_scatter-2.0.4-cp37-cp37m-linux_x86_64.whl
    pip install tools/torch_sparse-0.6.4-cp37-cp37m-linux_x86_64.whl
    pip install tools/torch_spline_conv-1.2.0-cp37-cp37m-linux_x86_64.whl
    pip install torch-geometric==1.5.0
    ```
4. 安装其他软件，`pip install -r requirements.txt`


## 运行说明

`cd neuroc_pygs`

1. 常规运行流程，`python base_experiment.py --model gcn --dataset pubmed --mode None --hidden_dims 32`

2. 基于流水线并行的图神经网络训练推理优化方法
``


## 注意事项

1. 当要运行某个py文件时，注意切换到该py文件的目录下再执行
如要执行`neuroc_pygs/sec5_memory/motivation_optimize.py`, 先执行`cd $ROOTPATH/neuroc_pygs/sec5_memory`, 再执行`python motivation_optimize.py`
> `$ROOTPATH`指工程目录，这里即`/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs`

2. 当修改一个其他文件时，先执行`python setup.py install --user`, 然后再运行该文件。

3. 默认数据集的位置是`/mnt/data/wangzhaokang/wangyunpan/data`
