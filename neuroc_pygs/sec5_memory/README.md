## 面向内存受限环境的图神经网络训练与推理流程优化方法

### 简要说明

图神经网络训练阶段，以GCN和GAT算法作为典型算法
评估了内存开销预测模型和训练阶段重采样比例的影响

### 目录结构

![](../../tools/文件关系图_sec5-6_memory.png)

```
exp5_thesis_figs: 最终的图像文件
out_motivation_data: 展示不同批次的内存波动问题的原始数据, 以及图5-6证明峰值内存开销与顶点数和边数之间的二元关

out_linear_model_datasets, out_linear_model_pth, out_linear_model_res: 线性内存开销预测模型相关目录
out_random_forest_datasets, out_random_forest_pth, out_random_forest_res: 随机森林内存开销模型相关目录

motivation.py: 常规处理流程，以及用来构建linear_model的样本
motivation_optimize.py: 内存受限处理流程

build_random_forest_datasets.py: 用来构建random_forest的样本
memory_model.py: 评估和保存内存开销模型
handle_overhead_data.py: 计算额外开销

prove_linear_plane.py: 证明峰值内存与边数和点数的二元线性关系（线性模型的成立条件）
prove_train_acc_memory_limit.py: 证明重采样对精度的影响

pics_thesis_motivation_edges.py: 图5-2,边数分布箱线图
pics_thesis_motivation_memory.py: 图5-3, 内存分布箱线图
pics_thesis_motivation_optimize.py: 如图5-8,5-9,5-10,5-11, 绘制内存开销模型执行后的结果
pics_thesis_train_acc_resampling.py: 图5-17, 绘制训练集精度与重采样比例的关系
pics_thesis_training.py: 图5-13, 训练阶段，使用重采样后，验证集上的精度的变化趋势
```
