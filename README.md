# optimize-pygs

#### Introduction

基于pyg的进一步优化工作。

#### Installation

1. `pip install -e .` 安装包; `python setup.py install --user`

#### Usage

CPU上运行
`python -m code.train --data_prefix ./data/<dataset_name> --train_config <path to train_config yml> --gpu -1`

GPU上运行
`python -m code.train --data_prefix ./data/<dataset_name> --train_config <path to train_config yml> --gpu <GPU number>`

2. openmp编译命令 `g++ Test.cpp -o omptest -fopenmp`