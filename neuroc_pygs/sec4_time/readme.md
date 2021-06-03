第四章 基于流水线并行的图神经网络训练推理流程优化

## 4.1 采样流程的流水线优化

### 核心文件

sampler/cuda_prefetcher.py

### 效果运行文件

sampling_batches.py: 测试优化前后的效果

### 效果整理文件

### 效果数据文件
查看

### 效果绘图文件
pics_thesis_batch_xxx(inferxxx).py: 绘图文件


## 4.2 训练与评估步骤的流水线优化

### 核心文件

epoch_opt_full_eval.py: 全数据方式评估进程程序
epoch_opt_full_train.py: 全数据方式训练进程程序

epoch_opt_sampling_eval.py: 采样的评估程序
epoch_opt_sampling_train.py: 采样的训练程序

本地文件共享：
exp_chekpoints

### 效果运行文件

epochs.py: 分批训练方式，优化效果运行文件
    - opt_epoch: 基于多线程方式实现流水线
    - epoch: 优化前

full_epochs.py: 全数据训练方式，优化效果运行文件。


epoch_base_sampling.py: 分批训练方式，"优化前"训练方式
epoch_base_full.py: 全数据方式，"优化前"训练方式

epochs_utils: 分批训练方式和全数据训练方式的train, test, infer函数文件

### 效果整理文件

### 效果结果文件


### 效果绘图文件

pics_thesis_xxx: 绘图文件

utils.py: 绘图的基本函数文件

## 4.3 叠加优化效果

### 核心文件

epoch_opt_full_eval.py: 全数据方式评估进程程序
epoch_opt_full_train.py: 全数据方式训练进程程序

epoch_opt_sampling_eval.py: 采样的评估程序
epoch_opt_sampling_train.py: 采样的训练程序


## 总目录结构

- 
- test_xx: 运行文件