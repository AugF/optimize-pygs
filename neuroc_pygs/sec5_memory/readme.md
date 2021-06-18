## 系统

motivation.py: 常规流程
- train(): 训练
- infer(): 推理
- train_loader(): 这里可以剥离出来

begin_train_loader: 准备train_loader的数据
run_one: 运行一次

motivation_predict.py: 内存受限处理流程

1. 内存开销预测模型
- 面向超参数确定的线性预测模型
- 面向超参数不确定的随机森林预测模型
> 以训练作为展示，注意这里需要统一为峰值内存，而不是膨胀内存
make_datasets.py: 构造数据集

> 两种内存开销模型的各种测试文件
- 准确率
- 准确率
- 额外开销（这个还需要额外计算）
> 这里的数据进行重新运行吧？

> 模型保存在哪？

2. 重采样机制
> 图5-13展示了中间的结果
（这里的文件需要重新寻找）
prove_train_acc_memory_limit.py

训练阶段，默认使用重采样后

3. 基于二分的超限子图上限预测方法


4. 基于度数和PageRank的超限子图剪枝方法


## 图片的来源
exp5_thesis_figs

- memory_model
exp_memory_training_gat_automl_mape_diff.png
exp_memory_training_gat_automl_r2_diff.png
exp_memory_training_gcn_automl_mape_diff.png
exp_memory_training_gcn_automl_r2_diff.png

- motivation
exp_memory_training_gat_cluster_motivation_diff.png
exp_memory_training_gat_cluster_motivation_edges_diff.png
gcn

- motivation_opt
exp_memory_training_gat_cluster_motivation_automl_mape_diff_v3.png
linear_model

- resampling
exp_memory_training_gat_linear_model_yelp_180_acc.png
loss.png
resampling_acc.png