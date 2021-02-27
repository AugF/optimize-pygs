import argparse
import torch
import os
import json
import traceback
import numpy as np
import os.path as osp
from neuroc_pygs.utils import get_dataset, gcn_norm, normalize, get_split_by_file, small_datasets
from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler
from neuroc_pygs.models import GCN, GGNN, GAT, GaAN
from neuroc_pygs.utils.evaluator import get_evaluator, get_loss_fn
from neuroc_pygs.configs import DATASET_METRIC, dataset_root, ALL_MODELS, EXP_DATASET, MODES, EXP_RELATIVE_BATCH_SIZE, PROJECT_PATH
from tabulate import tabulate


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pubmed', help="dataset: [flickr, com-amazon, reddit, com-lj,"
                        "amazon-computers, amazon-photo, coauthor-physics, pubmed]")
    parser.add_argument('--model', type=str, default='gcn',
                        help="gnn models: [gcn, ggnn, gat, gaan]")
    parser.add_argument('--epochs', type=int, default=9,
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
                        default='cuda:1', help='[cpu, cuda:id]')
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
    parser.add_argument('--relative_batch_size', type=float,
                            help='number of cluster partitions per batch') # relative batch size
    parser.add_argument('--infer_batch_size', type=int, default=1024,
                            help='number of cluster partitions per batch') # inference batch size
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
    # for log
    parser.add_argument('--log_batch', type=bool,
                        default=False, help='log batch')
    parser.add_argument('--log_epoch', type=bool,
                        default=False, help='log epoch')
    parser.add_argument('--log_batch_dir', type=str,
                        default=None, help='log epoch dir')
    parser.add_argument('--log_epoch_dir', type=str,
                        default=None, help='log epoch dir')
    # for train
    parser.add_argument('--log_step', type=int,
                        default=1, help='log step')
    parser.add_argument('--eval_step', type=int,
                        default=1, help='eval step')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/checkpoints', help='eval step')
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


def build_train_loader(args, data, Cluster_Loader=ClusterLoader, Neighbor_Loader=NeighborSampler):
    if args.relative_batch_size:
        args.batch_size = int(data.x.shape[0] * args.relative_batch_size)
        args.batch_partitions = int(args.cluster_partitions * args.relative_batch_size)
    if args.mode == 'cluster':
        cluster_data = ClusterData(data, num_parts=args.cluster_partitions, recursive=False,
                                   save_dir=args.cluster_save_dir)
        train_loader = Cluster_Loader(cluster_data, batch_size=args.batch_partitions, shuffle=True,
                                     num_workers=args.num_workers, pin_memory=args.pin_memory)
    elif args.mode == 'graphsage':
        train_loader = Neighbor_Loader(data.edge_index, node_idx=None,
                                       sizes=[25, 10], batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, pin_memory=args.pin_memory)
    return train_loader


def build_subgraphloader(args, data, Neighbor_Loader=NeighborSampler):
    if args.relative_batch_size:
        args.batch_size = int(data.x.shape[0] * args.relative_batch_size)
        args.batch_partitions = int(args.cluster_partitions * args.relative_batch_size)

    if args.infer_layer:
        subgraph_loader = Neighbor_Loader(data.edge_index, sizes=[-1], batch_size=args.infer_batch_size,
                                          shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    else:
        subgraph_loader = Neighbor_Loader(data.edge_index, sizes=[-1] * args.layers, batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    return subgraph_loader


def build_loader(args, data, Cluster_Loader=ClusterLoader, Neighbor_Loader=NeighborSampler):
    train_loader = build_train_loader(args, data)
    subgraph_loader = build_subgraphloader(args, data)
    return train_loader, subgraph_loader


def build_optimizer(args, model):
    optimizer = torch.optim.Adam([
    dict(params=model.convs[i].parameters(),
            weight_decay=args.weight_decay if i == 0 else 0)
    for i in range(1 if args.model == "ggnn" else args.layers)], lr=args.lr)  # Only perform weight-decay on first convolution, 参考了pytorch_geometric中的gcn.py的例子: https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gcn.py
    return optimizer


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

    # step2 set loss_fn and evaluator
    model.set_loss_fn(get_loss_fn(DATASET_METRIC[args.dataset]))
    model.set_evaluator(get_evaluator(DATASET_METRIC[args.dataset]))
    return model


def build_model_optimizer(args, data):
    model = build_model(args, data)
    optimizer = build_optimizer(args, model)
    return model, optimizer


def prepare_trainer(**kwargs):
    args = get_args()
    for key, value in kwargs.items():
        if key in args.__dict__.keys():
            args.__setattr__(key, value)
    print(args)
    data = build_dataset(args)
    train_loader, subgraph_loader = build_loader(args, data)
    model, optimizer = build_model_optimizer(args, data)  
    return data, train_loader, subgraph_loader, model, optimizer, args


def run(func, runs=3, path="run.out", model='gcn', dataset='pubmed', mode='cluster', relative_batch_size=None):
    if not isinstance(model, list):
        model = [model]
    if not isinstance(dataset, list):
        dataset = [dataset]
    if not isinstance(mode, list):
        mode = [mode]
    if not isinstance(relative_batch_size, list):
        relative_batch_size = [relative_batch_size]
    run_all(func, runs, path, exp_datasets=dataset, exp_models=model, exp_modes=mode, exp_relative_batch_sizes=relative_batch_size)


def run_all(func, runs=3, path='run_all.out', exp_datasets=EXP_DATASET, exp_models=ALL_MODELS, exp_modes=MODES, exp_relative_batch_sizes=EXP_RELATIVE_BATCH_SIZE):
    real_path = osp.join(PROJECT_PATH, 'log', path)
    if osp.exists(real_path):
        tab_data = json.load(open(real_path))
    else: 
        tab_data, success_tabs = {}, []

    # 读取上次已经成功的数据
    fp = open(real_path + '.keys', 'a+')
    success_tabs = fp.read().strip().split('\n')

    args = get_args()
    for exp_data in exp_datasets:
        args.dataset = exp_data
        data = build_dataset(args) # step1 data
        print('build data success!')
        for exp_model in exp_models:
            args.model = exp_model
            data = data.to('cpu')
            model, optimizer = build_model_optimizer(args, data) # step2 build model
            print('build model success!')
            for exp_relative_batch_size in exp_relative_batch_sizes:
                args.relative_batch_size = exp_relative_batch_size # step3 set batch size
                for exp_mode in exp_modes:
                    data = data.to('cpu')
                    train_loader, subgraph_loader = build_loader(args, data) # step4 loader
                    print('build loader success!')
                    file_name = '_'.join([exp_data, exp_model, exp_mode, str(exp_relative_batch_size)])
                    print(file_name)
                    if file_name in success_tabs: # 避免重复实验
                        continue
                    try:
                        base_times, opt_times, ratios = [], [], []
                        for _ in range(runs):
                            base_time, opt_time, ratio = func(data, train_loader, subgraph_loader, model, optimizer, args) # fit
                            base_times.append(base_time)
                            opt_times.append(opt_time)
                            ratios.append(ratio)
                        tab_data[file_name] = [file_name, np.mean(base_times), np.mean(opt_times), np.mean(ratios)]
                        print(tab_data[file_name])
                        fp.write(file_name + '\n') # 实时写入文件
                    except Exception as e:
                        print(e.args)
                        print("======")
                        print(traceback.format_exc())

                    torch.cuda.empty_cache()
                    del train_loader, subgraph_loader # 内存使用需要慎重
            del model
        del data

    fp.close() # 关闭文件
    json.dump(tab_data, open(real_path, 'w'))
    print(tabulate(list(tab_data.values()), headers=["(mode, relative_batch_size, model, data)", "Base Time(s)", "Optmize Time(s)", "Ratio(%)"],
            tablefmt="github"))
    


if __name__ == "__main__":
    import sys
    sys.argv = [sys.argv[0], '--dataset', 'amazon', '--log_batch', 'True']
    print(get_args())