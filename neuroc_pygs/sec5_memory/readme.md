面向内存受限环境的图神经网络训练与推理流程优化

## 说明

图神经网络训练阶段，以GCN和GAT算法作为典型算法
评估了内存开销预测模型和训练阶段重采样比例的影响

## 目录结构
exp_automl_datasets_diff
exp_figs
exp_log_diff
exp_motivation_diff
exp_train_figs
exp_res


out_motivation_data: 
- 1. 展示不同批次的训练的题
- 2. 图5-6证明峰值内存开销与顶点数和边数之间的二元关系


exp5_thesis_figs

线性模型
- 1. 构建数据集，保存在out_linear_model_datasets; (motivation.py)
- 2. out_linear_model_pth: 训练好的模型文件
- 3. out_linear_model_res: 使用线性模型后保存的文件结果

随机森林
- 1. 构架数据集，保存在out_random_forest_datasets (build_random_forest_datasets.py)
- 2. out_random_forest_pth: 训练好的模型文件
- 3. out_random_forest_res: 使用随机森林后保存的文件

motivation_optimize.py: 使用内存开销模型后的结果

handle_overhead_data.py: 计算额外开销
memory_model.py: 评估和保存内存开销模型

prove_linear_plane.py: 图5-6证明，用于收集文件
prove_train_acc_memory_limit.py: 证明重采样对精度的影响


pics_thesis_motivation_edges.py: 图5-2, 边数分布箱线图
pics_thesis_motivation_memory.py: 图5-3, 内存分布箱线图
pics_thesis_motivation_optimize.py: 如图5-8,5-9,5-10,5-11, 绘制内存开销模型执行后的结果
pics_thesis_train_acc_resampling.py: 图5-17, 绘制训练集精度与重采样比例的关系
pics_thesis_training.py: 图5-13,, 训练阶段，使用重采样后，验证集上的精度的变化趋势

configs.py: 算法默认的参数配置