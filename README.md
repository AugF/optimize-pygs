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

3. cython编译命令 `python setup.py build_ext --inplace`

4. pybind11编译命令
> g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` example.cpp -o example`python3-config --extension-suffix` -lpthread
> g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` call_python.cpp -o call_python`python3-config --extension-suffix` -fopenmp