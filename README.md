## 简介

optimize_pygs是基于PyTorch Geometric(PyG 1.5.0)的工作。
针对基于采样的图神经网络训练与推理中存在的两大性能问题：
1. 采样流程额外开销大、评估步骤耗时较长
2. 不同批次间的内存波动大，某些极大值可能导致内存溢出风险。

针对以上两大性能问题，提出了两种流程优化方法:
1. 基于流水线并行的图神经网络训练与推理流程优化方法(neuroc_pygs/sec4_time)
2. 面向内存受限环境的图神经网络训练与推理流程优化方法(neuroc_pygs/sec5_memory, neuroc_pygs/sec6_cutting)

## 组织架构

- models: 模型文件
- samplers: 采样方法
- sec4_time: 
- sec5_memory:
- sec6_cutting:
- utils: 辅助文件

pyg15_main_sampling_graph_info.py
pyg15_main_sampling.py
pyg15_main.py