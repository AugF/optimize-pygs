import torch
from torch_sparse import SparseTensor, sample_adj
# from torch_sparse.tensor import SparseTensor


def test_saint_subgraph():
    row = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3, 4])
    col = torch.tensor([1, 2, 0, 2, 0, 1, 3, 2, 4, 3])
    adj = SparseTensor(row=row, col=col)
    node_idx = torch.tensor([0, 1, 2])

    adj, edge_index = adj.saint_subgraph(node_idx)  # 通过选点
    print('saint', adj, edge_index, node_idx)


def test_sample_adj():
    row = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 4, 4, 5, 5])
    col = torch.tensor([1, 2, 3, 0, 2, 0, 1, 4, 5, 0, 2, 5, 2, 4])
    value = torch.arange(row.size(0))
    adj_t = SparseTensor(row=row, col=col, value=value, sparse_sizes=(6, 6))
    print('adj', adj_t)

    out, n_id = sample_adj(adj_t, torch.arange(2, 6), num_neighbors=-1)
    print('sample', out, n_id)
    assert n_id.tolist() == [2, 3, 4, 5, 0, 1]

    row, col, val = out.coo()
    # assert row.tolist() == [0, 0, 0, 0, 1, 2, 2, 3, 3]
    # assert col.tolist() == [2, 3, 4, 5, 4, 0, 3, 0, 2]
    # assert val.tolist() == [7, 8, 5, 6, 9, 10, 11, 12, 13]

    out, n_id = sample_adj(adj_t, torch.arange(2, 6), num_neighbors=2,
                           replace=True)
    print(out, n_id)
    # assert out.nnz() == 8

    out, n_id = sample_adj(adj_t, torch.arange(2, 6), num_neighbors=2,
                           replace=False)
    # assert out.nnz() == 7  # node 3 has only one edge...
    print(out, n_id)


# test_saint_subgraph()
test_sample_adj()
