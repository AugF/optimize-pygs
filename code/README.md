
## 贡献1：性能瓶颈分析 


## 贡献2: 基于采样的优化
Motivation:
- 对于单机多GPU和分布式的场景，落到每个GPU上，Sampler这块的具体做法也是和我们类似的
- 实际中往往有资源有限的场景，特别是遇到图神经网络以及更复杂的网络，运行时间过长始终是巨大的问题

目标: 对训练流程进行优化和改进
输出：一套标准化流程


### 优化点1: 流水线技术优化程序执行流程

#### 思路
1. 采样overlap这个点，尝试优化思路
    > 思路1： 采样看作生成者，To和Training看作是消费者;  
    >   - 问题1：To和Trainning需不需要异步，即是否需要提前将内存放到GPU上去；可以考虑每次都将To删除
    > 不动精度
    > 代码：涉及到的问题，中间结果如何保存的问题;
    >   - 流水线： 最小的内存占用
    >   - 所有的都
    >  - evaluation在cpu端进行
    >  - 实际上Dataloader的sampler已经在毫秒级别了
    > 要证明batch_size大了的好处才是，这个有证据可以表明的
    >   - 之前gnn的证明是否是有效

2. 主存都放不下的情况下如何采样?

#### 计划

[x] 查看现有GNN系统是否已经采用了贡献2的想法
    [x] 阅读“大规模图神经网络系统综述", 没有涉及该部分内存
    [x] 查看pyg, dgl官方是否已经做过这部分内容; 总结：现有的图神经网络系统都没有用到
        [x] pyg没有用到
        [x] ogb没有用到
        [x] dgl没有用到
    > 认为是可以做的点
[x] 整体思路
    [ ] 
[ ] 安装


- mode: (train, evaluation, test, optimizer, sampler, **args)
    - 固定epoch
        - input: *args, epoch
    - early_stopping
        - input: **args, hot epochs, patience 
        - output: 

- 核心：
    - training和evaluation在不同GPU上，数据保存在内存中，数据存储到文件
    - training在CPU, evaluation在GPU
    - sampler和GPU

#### 相关技术和知识点

- python多线程
- to
- 数据存储

train_loader: 每一运行每一个batch的数据都不一样


## 贡献3: 基于GPU的内存保护

motivation:
1. 防止中间OOM带来的问题，一旦崩溃便难以处理
2. Inference阶段采样出来的子图是变化的，这样来说是有意义的
3. 考虑如何才可以，将这一套做得有意义

