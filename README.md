# optimize-pygs

#### Introduction

基于pyg的进一步优化工作。

#### Installation

1. `python setup.py install --user`

不起作用时，删除本地库中的包
/home/wangzhaokang/.local/lib/python3.7/site-packages/code-0.1.0-py3.7.egg/neuroc_pygs/utils/datasets.py

#### Usage

CPU上运行
`python -m code.train --data_prefix ./data/<dataset_name> --train_config <path to train_config yml> --gpu -1`

GPU上运行
`python -m code.train --data_prefix ./data/<dataset_name> --train_config <path to train_config yml> --gpu <GPU number>`

2. openmp编译命令 `g++ Test.cpp -o omptest -fopenmp`

3. cython编译命令 `python setup.py build_ext --inplace`

4. pybind11编译命令
> g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` example.cpp -o example`python3-config --extension-suffix` -lpthread
> g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` call_python.cpp -o call_python`python3-config --extension-suffix` -fopenmp

### 待做

[ ] 修正neuroc_pygs和graphsaint_pygs的数据集的问题

### 计划

脑袋必须清醒!

- 实验数据集
```
'pubmed', 'amazon-photo', 'amazon-computers', 'ppi', 'flickr', 'reddit', 'yelp', 'amazon'
```
model.to(non_blocking=True)
[参考计时](https://blog.csdn.net/handsomeasme/article/details/104143093)

> 统计最好情况的时间消耗
#### opt_train计划

任务: 先让总时间提上去再说

[tensor.to()官方文档](https://pytorch.org/docs/stable/tensors.html?highlight=#torch.Tensor.to)

[data_prefetch](https://www.cnblogs.com/pprp/p/14199865.html#5-data-prefetch)

[torch.cuda.stream](https://codesuche.com/python-examples/torch.cuda.stream/)

[如何给你的DataLoader打鸡血](https://zhuanlan.zhihu.com/p/66145913)
[训练加速](https://zhuanlan.zhihu.com/p/80695364)
[Tools for easy mixed precision and distributed training in Pytorch](https://github.com/NVIDIA/apex)