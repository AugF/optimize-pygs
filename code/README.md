
## 第四章

分为基于采样的优化和基于流水线的优化

### 目标

### TODO
[x] 查看现有GNN系统是否已经采用了贡献2的想法
    [x] 阅读“大规模图神经网络系统综述", 没有涉及该部分内存
    [x] 查看pyg, dgl官方是否已经做过这部分内容; 总结：现有的图神经网络系统都没有用到
        [x] pyg没有用到
        [x] ogb没有用到
        [x] dgl没有用到
[ ] 采样的流水线技术实现
    [ ] optimize batch: ogbn_products_sage;
        - train
        - inference
    [ ] optimize sampler
        > 先使用，再考虑优化
    [ ] optimize epoch
    [ ] 进一步考虑使用什么实现流水线
        [ ] python协程
        [ ] 分布式Ray
        [ ] C++多线程库,cuda stream + pybind   
[ ] 训练的流水线技术实现
[ ] 采样的优化实现
[ ] 典型场景下，上述三个技术的实现
[ ] 综合使用三个策略下的分析，说明哪些策略对哪些算法更有效果

## 第五章

### 目标
