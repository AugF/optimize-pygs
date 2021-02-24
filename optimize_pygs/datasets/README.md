添加一个数据集mydataset的步骤

1. 在`optimize_pygs.global_configs.dataset_root`指定的目录下，创建以下文件:
> `io.py`文件提供了创建随机图的方法(snap-standford生成随机图)
```
dataset_root
    mydataset
        raw
            adj_full.npz
            adj_train.npz (Optional)
            feats.npy
            class_map.json
            role.json
```

`adj_full.npz`: a sparse matrix in CSR format, stored as a `scipy.sparse.csr_matrix`. The shape is N by N. Non-zeros in the matrix correspond to all the edges in the full graph. It doesn't matter if the two nodes connected by an edge are training, validation or test nodes. For unweighted graph, the non-zeros are all 1.

`adj_train.npz`: a sparse matrix in CSR format, stored as a `scipy.sparse.csr_matrix`. The shape is also N by N. However, non-zeros in the matrix only correspond to edges connecting two training nodes. The graph sampler only picks nodes/edges from this `adj_train`, not `adj_full`. Therefore, neither the attribute information nor the structural information are revealed during training. Also, note that only aN rows and cols of `adj_train` contains non-zeros. See also issue #11. For unweighted graph, the non-zeros are all 1.

`role.json`: a dictionary of three keys. Key `'tr'` corresponds to the list of all training node indices. Key `va` corresponds to the list of all validation node indices. Key `te` corresponds to the list of all test node indices. Note that in the raw data, nodes may have string-type ID. You would need to re-assign numerical ID (0 to N-1) to the nodes, so that you can index into the matrices of adj, features and class labels.

`class_map.json`: a dictionary of length N. Each key is a node index, and each value is either a length C binary list (for multi-class classification) or an integer scalar (0 to C-1, for single-class classification).

`feats.npy`: a `numpy` array of shape N by F. Row i corresponds to the attribute vector of node i.

2. 创建一个该数据集的类
```python
# optimize_pygs/datasets/mydataset.py
@register_dataset("mydataset")
class MyDatasetDataset(CustomDataset):
    def __init__(self):
        dataset = "mydataset"
        super(MyDatasetDataset, self).__init__(dataset_root, dataset)

    def get_evaluator(self): # 指定evaluator
        return multilabel_f1

    def get_loss_fn(self): # 指定loss_fn
        return bce_with_logits_loss
```

3. 在`__init__.py`文件中注册
```
SUPPORTED_DATASETS = {
    ...
    'mydataset': 'optimize_pygs.datasets.mydatasetdata', # added
    ...
}    
```

4. 测试
```python
from optimize_pygs.datasets import build_dataset
from optimize_pygs.utils import build_args_from_dict

data = build_dataset(build_args_from_dict({'dataset': 'mydataset'}))
print(data[0])
```

## TODO: 开放设置train_mask, val_mask, test_mask的接口