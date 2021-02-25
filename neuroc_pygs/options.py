import argparse
import torch
import os
import numpy as np
import os.path as osp
from neuroc_pygs.utils import get_dataset, gcn_norm, normalize, get_split_by_file, small_datasets
from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler
from neuroc_pygs.models import GCN, GGNN, GAT, GaAN
from neuroc_pygs.utils.evaluator import get_evaluator, get_loss_fn
from neuroc_pygs.configs import DATASET_METRIC, dataset_root


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pubmed', help="dataset: [flickr, com-amazon, reddit, com-lj,"
                        "amazon-computers, amazon-photo, coauthor-physics, pubmed]")
    parser.add_argument('--model', type=str, default='gcn',
                        help="gnn models: [gcn, ggnn, gat, gaan]")
    parser.add_argument('--epochs', type=int, default=11,
                        help="epochs for training")
    parser.add_argument('--layers', type=int, default=2,
                        help="layers for hidden layer")
    parser.add_argument('--hidden_dims', type=int,
                        default=64, help="hidden layer output dims")
    parser.add_argument('--heads', type=int, default=8,
                        help="gat or gaan model: heads")
    # head_dims * heads = hidden_dims
    parser.add_argument('--head_dims', type=int, default=8,
                        help="gat model: head dims")
    # d_v * heads = hidden_dims?
    parser.add_argument('--d_v', type=int, default=8,
                        help="gaan model: vertex's dim")
    parser.add_argument('--d_a', type=int, default=8,
                        help="gaan model: each vertex's dim in edge attention")  # d_a = head_dims
    parser.add_argument('--d_m', type=int, default=64,
                        help="gaan model: gate: max aggregator's dim, default=64")
    parser.add_argument('--x_sparse', action='store_true',
                        default=False, help="whether to use data.x sparse version")

    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--device', type=str,
                        default='cuda:2', help='[cpu, cuda:id]')
    parser.add_argument('--cpu', action='store_true',
                        default=False, help='use cpu, not use gpu')
    parser.add_argument('--lr', type=float, default=0.01,
                        help="adam's learning rate")
    parser.add_argument('--weight_decay', type=float,
                        default=0.0005, help="adam's weight decay")
    parser.add_argument('--no_record_shapes', action='store_false',
                        default=True, help="nvtx or autograd's profile to record shape")
    parser.add_argument('--json_path', type=str, default='',
                        help="json file path for memory")
    parser.add_argument('--infer_json_path', type=str, default='',
                        help="inference stage: json file path for memory")
    parser.add_argument('--mode', type=str, default='cluster',
                        help='sampling: [cluster, graphsage]')
    parser.add_argument('--batch_size', type=int,
                        default=1024, help='batch size')
    parser.add_argument('--batch_partitions', type=int, default=20,
                        help='number of cluster partitions per batch')
    parser.add_argument('--cluster_partitions', type=int,
                        default=1500, help='number of cluster partitions')
    parser.add_argument('--num_workers', type=int, default=40,
                        help='number of Data Loader partitions')
    parser.add_argument('--infer_layer', type=bool,
                        default=True, help='Choose how to inference')
    parser.add_argument('--pin_memory', type=bool,
                        default=True, help='pin_memory')
    args = parser.parse_args()
    args.gpu = not args.cpu and torch.cuda.is_available()
    args.flag = not args.json_path == ''
    args.infer_flag = not args.infer_json_path == ''

    os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
    # 0. set manual seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed(args.seed)
    
    args.device = torch.device(args.device if args.gpu else 'cpu')
    return args


def build_dataset(args):
    # step1 build data
    dataset_info = args.dataset.split('_')
    if dataset_info[0] in small_datasets and len(dataset_info) > 1:
        args.dataset = dataset_info[0]

    dataset = get_dataset(args.dataset, normalize_features=True)
    data = dataset[0]

    # add train, val, test split
    if args.dataset in ['amazon-computers', 'amazon-photo', 'coauthor-physics']:
        file_path = osp.join(
            dataset_root, args.dataset + "/raw/role.json")
        data.train_mask, data.val_mask, data.test_mask = get_split_by_file(
            file_path, data.num_nodes)

    num_features = dataset.num_features
    if dataset_info[0] in small_datasets and len(dataset_info) > 1:
        file_path = osp.join(dataset_root,
                             "data/feats_x/" + '_'.join(dataset_info) + '_feats.npy')
        if osp.exists(file_path):
            data.x = torch.from_numpy(np.load(file_path)).to(
                torch.float)  # 因为这里是随机生成的，不考虑normal features
            num_features = data.x.size(1)

    args.num_features, args.num_classes, args.cluster_save_dir = num_features, dataset.num_classes, dataset.processed_dir
    return data


def build_loader(args, data):
    if args.infer_layer:
        subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1], batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    else:
        subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1] * args.layers, batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    if args.mode == 'cluster':
        cluster_data = ClusterData(data, num_parts=args.cluster_partitions, recursive=False,
                                   save_dir=args.cluster_save_dir)
        train_loader = ClusterLoader(cluster_data, batch_size=args.batch_partitions, shuffle=True,
                                     num_workers=args.num_workers, pin_memory=args.pin_memory)
    elif args.mode == 'graphsage':
        train_loader = NeighborSampler(data.edge_index, node_idx=None,
                                       sizes=[25, 10], batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, pin_memory=args.pin_memory)
    return train_loader, subgraph_loader


def build_model(args, data):
    # step1 build model
    if args.model == 'gcn':
        # 预先计算edge_weight出来
        norm = gcn_norm(data.edge_index, data.x.shape[0])
        model = GCN(
            layers=args.layers,
            n_features=args.num_features, n_classes=args.num_classes,
            hidden_dims=args.hidden_dims, gpu=args.gpu, flag=args.flag, infer_flag=args.infer_flag,
            device=args.device, cached_flag=False, norm=norm
        )
    elif args.model == 'gat':
        model = GAT(
            layers=args.layers,
            n_features=args.num_features, n_classes=args.num_classes,
            head_dims=args.head_dims, heads=args.heads, gpu=args.gpu,
            flag=args.flag, infer_flag=args.infer_flag, sparse_flag=args.x_sparse, device=args.device,
        )
    elif args.model == 'ggnn':
        model = GGNN(
            layers=args.layers,
            n_features=args.num_features, n_classes=args.num_classes,
            hidden_dims=args.hidden_dims, gpu=args.gpu, flag=args.flag,
            infer_flag=args.infer_flag, device=args.device
        )
    elif args.model == 'gaan':
        model = GaAN(
            layers=args.layers,
            n_features=args.num_features, n_classes=args.num_classes,
            hidden_dims=args.hidden_dims,
            heads=args.heads, d_v=args.d_v,
            d_a=args.d_a, d_m=args.d_m, gpu=args.gpu,
            flag=args.flag, infer_flag=args.infer_flag, device=args.device
        )

    # step2 optimizer
    optimizer = torch.optim.Adam([
        dict(params=model.convs[i].parameters(),
             weight_decay=args.weight_decay if i == 0 else 0)
        for i in range(1 if args.model == "ggnn" else args.layers)], lr=args.lr)  # Only perform weight-decay on first convolution, 参考了pytorch_geometric中的gcn.py的例子: https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gcn.py

    # step3 set loss_fn and evaluator
    model.set_loss_fn(get_loss_fn(DATASET_METRIC[args.dataset]))
    model.set_evaluator(get_evaluator(DATASET_METRIC[args.dataset]))
    return model, optimizer


if __name__ == "__main__":
    print(get_args())