import numpy as np
import snap # in linux, need py35, py36 or py37
import json
import os
import scipy.sparse as sp

def create_random_dataset(
    dataset_name, nodes, edges,
    features=32, classes=10, split_train=0.70, split_val=0.15,
    root="/mnt/data/wangzhaokang/wangyunpan/datasets"):
    """
    root: dataset所存储的目录
    """
    Rnd = snap.TRnd()
    print("nodes={}, edges={}".format(nodes, edges))
    graph = snap.GenRMat(nodes, edges, .6, .1, .15, Rnd)
    raw_dir = root + "/" + dataset_name + "/raw"
    print(raw_dir)
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)

    print("get adj_full.npz...")
    edges_list = []
    for e in graph.Edges():
        r, c = e.GetSrcNId(), e.GetDstNId()
        edges_list.append((r, c))
        edges_list.append((c, r))
    
    edges_list = set(edges_list)
    row = [i[0] for i in edges_list]
    col = [i[1] for i in edges_list]
    print(edges, len(row))
    f = sp.csr_matrix(([1] * len(row), (row, col)), shape=(nodes, nodes)) # directed -> undirected, edges*2
    np.savez(raw_dir + "/adj_full", data=f.data, indptr=f.indptr, indices=f.indices, shape=f.shape)
    gen_graph(raw_dir, nodes, edges, features, classes, split_train, split_val)
        
# https://snap.stanford.edu/snappy/doc/reference/GenRMat.html
def gen_graph(raw_dir, nodes, edges, features=32, classes=10, split_train=0.70, split_val=0.15, seed=1):
    np.random.seed(seed)
    print("get class_map.json...")
    # 1. class_map.json
    class_map = {i: np.random.randint(0, classes) for i in range(nodes)}
    with open(raw_dir + "/class_map.json", "w") as f:
        json.dump(class_map, f)
    
    print("get role.json...")
    # 2. role.json: tr, va, te
    idx = np.arange(nodes)
    np.random.shuffle(idx)
    tr, va = int(nodes * split_train), int(nodes * (split_train + split_val))
    role = {'tr': idx[: tr].tolist(),
            'va': idx[tr: va].tolist(),
            'te': idx[va:].tolist()
            }

    with open(raw_dir + "/role.json", "w") as f:
        json.dump(role, f)
    
    print("get feats.npy...")
    # 3. feats.npy
    feats = np.random.randn(nodes, features)
    np.save(raw_dir + "/feats", feats)
    