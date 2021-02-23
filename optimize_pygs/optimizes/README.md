## 时间优化
### 1. optimize sampler(cluster)

#### `optimize_cluster_graphsaint_data.log`

目的：探究数据集规模对优化效果的影响; 探究优化模块和整个模块优化效果的一致性; 探究loader的多线程对优化效果的影响; 探究batch size对优化效果的影响


#### `optimize_custer_neuroc_data.log`


### 2. optimize batch


### 3. optimize epoch


```

```

https://docs.ray.io/en/latest/actors.html

python程序 -> c_sampler, 难度很大

分布式框架: ray还可以用来加速

测试model.to进行异步的方法; 
从某种意义上说应该不是io受限

sampler寻思可不可以通过换语言写实现加速

## 内存优化

### 1. motivation验证

大数据集下经常在大的BatchSize下cash掉
内存预测模型: 可以用来更好的预判内存使用情况

师兄的观点:
在Inference阶段效果更为明显，可以有效地减少时间
