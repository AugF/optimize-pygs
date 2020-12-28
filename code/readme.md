
## 贡献2: 基于采样的优化


### 优化点1： sampling, data_transferring, training的overlap

#### 思路
1. 采样overlap这个点，尝试优化思路
    > 思路1： 采样看作生成者，To和Training看作是消费者;  
    >   - 问题1：To和Trainning需不需要异步，即是否需要提前将内存放到GPU上去；可以考虑每次都将To删除
    >   - 问题2:  
2. 

#### 计划
[ ] 验证torch.cuda的data_transferring和training的接口, [参考]()
    [ ] torch.cuda(non_blocking=True): 
[ ] 验证python的multiprocessing机制:
    > 

### 优化点2： graphsage采样技术的优化