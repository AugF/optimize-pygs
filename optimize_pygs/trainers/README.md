## Introduction

## Usage

## TODO

时态

在未来，这仍有一些本文没有覆盖的方向值得我们去关注。
In the future, we will focus on but not limited to the following directions.
1) 单机多卡/分布式训练中的性能瓶颈. performance bottlenecks in training and inference with multi-GPUs or distributed environments.

For massive data in the real world, multi-GPUs and distributed training are often the more common methods[1,2,3].
These methods will inevitably introduce overheads such as communication between GPUs (inter-machines). How these overheads affect performance bottlenecks is worthy to focus on.
对于真实世界的海量数据，多卡和分布式训练往往是更有效的处理方式。它们的训练必然会引入卡间（机器间）的通信等开销，这些开销如何影响性能瓶颈是未来值得做的工作。
2) 不同编程模型之间的性能瓶颈是否一致。impacts of difference programming models on performance bottlenecks.

Our work is based on the mainstream programming model messaging framework. However, some graph neural network computing systems also proposed some new processing models. For example, NeuGraph[1] proposed the SAGA framework, which includes four stages: Scatter, ApplyEdge, Gather, and ApplyVertex; EuGN [4] provides an edge-centric processing model. Whether different programming models will lead to changes in performance bottlenecks is worth discussing.

我们的分析是基于主流的编程模型——消息-传递框架，然而一些图神经网络计算系统还提出了一些新的处理模型，如NeuGraph提出的SAGA框架，包含了四个阶段: Scatter, ApplyEdge, Gather, ApplyVertex; EuGN[1]提供了一种以边为中心的处理模型等等。编程模型的差异是否会带来性能瓶颈的变化是值得讨论的工作.

3) impacts of spatial-temporal graph on performance bottlenecks.
动态图出现中在交通预测，人的行为识别和行人检测等多种应用中。
如何从动态图中学习潜在的表示是越来越重要的工作。
关于动态图的GNN工作也相继提出，如[1],[2],[3]。动态图是否会影响性能瓶颈值得我们关注
Spatial-temporal graphs appear in a variety of applications such as traffic prediction, human behavior recognition, and pedestrian detection.
Learning hidden patterns from spatial-temporal graphs becomes increasingly important.
Many GNNs are also proposed([1], [2], [3]). Whether spatial-temporal graphs will affect performance is worthy of our attention.


Spatial–Temporal Graph Neural Networks: These aim
to learn hidden patterns from spatial–temporal graphs, which becomes increasingly important in a variety of applications, such as traffic speed forecasting [72], driver maneuver anticipation [73], and human action recognition [75]. The key idea of STGNNs is to consider spatial dependence and temporal dependence at the same time. Many current approaches inte- grate graph convolutions to capture spatial dependence with RNNs or CNNs to model temporal dependence. Fig. 2(d) illustrates an STGNN for spatial–temporal graph forecasting。

This work highlights a number of directions for future
work. One direction is to develop algorithms that consider the factors indicated above: (i) impacts of workload and graph characteristics on computational load balance, and (ii) impacts of workload execution skew on the workload performance. Another direction is to study the appropriate scale-out factor given a particular graph and workload char- acteristics. This is a general problem but is important in this context since some of the algorithms are sensitive to the communication-to-computation ratio

In the future, we will focus on but not limited to the following directions: 1) GNN for edge-level and subgraph-level embeddings; 2) More execution optimizations, such as co-location of computation variables in GNN with graph data to reduce the cross network traffic, introduction of new gradient optimization to leverage the trait of GNN to speed up the distributed training without accuracy loss, and better assignment of the workers in multi-GPU architectures; 3) Early-stop mechanism, which can help to terminate training tasks earlier when no promising results can generate.