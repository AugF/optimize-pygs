import torch
import numpy as np
import os.path as osp
from optimize_pygs.global_configs import NEUROC_DATASET, SAINT_DATASET, neuroc_dataset_root
from optimize_pygs.options import get_default_args
from optimize_pygs.utils.pyg15_utils import gcn_norm, get_split_by_file
from optimize_pygs.datasets import build_dataset
from optimize_pygs.models import build_model
from optimize_pygs.loaders import build_sampler_from_name
from optimize_pygs.loaders.configs import TRAIN_CONFIG, INFER_CONFIG
from optimize_pygs.trainers import build_trainer


def experiment(model, dataset, sampler, infer_layer=False, **kwargs):
    args = get_default_args(model, dataset, sampler=sampler, infer_layer=infer_layer, **kwargs)
    # step1. load dataset
    dataset = build_dataset(args)  # dataset_args
    data = dataset[0]

    # add train, val, test split
    if args.dataset in ['amazon-computers', 'amazon-photo', 'coauthor-physics']:
        file_path = osp.join(neuroc_dataset_root, args.dataset + "/raw/role.json")
        data.train_mask, data.val_mask, data.test_mask = get_split_by_file(file_path, data.num_nodes)

    print(data)
    # step2. load model
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes

    if model == "pyg15_gcn":
        args.norm = gcn_norm(data.edge_index, data.x.shape[0])
    model = build_model(args)  # args
    model.set_loss_fn(dataset.get_loss_fn())
    model.set_evaluator(dataset.get_evaluator())
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.manual_seed(args.seed)
    
    # step3 load loader
    train_loader = build_sampler_from_name(args.sampler, data=data, save_dir=dataset.processed_dir, num_parts=args.num_parts,
                                           batch_partitions=args.batch_partitions, batch_size=args.batch_size, num_workers=args.num_workers,
                                           **TRAIN_CONFIG[args.sampler])

    if args.infer_layer:
        infer_loader = build_sampler_from_name(args.infer_sampler, data=data, sizes=[-1],
                                           batch_size=args.infer_batch_size, num_workers=args.num_workers,
                                           **INFER_CONFIG[args.infer_sampler])
    else:
        infer_loader = build_sampler_from_name(args.infer_sampler, data=data, sizes=[-1] * args.num_layers,
                                              batch_size=args.infer_batch_size, num_workers=args.num_workers,
                                              **INFER_CONFIG[args.infer_sampler])

    # step4 training
    trainer = build_trainer(args)
    model, data = model.to(trainer.device), data.to(trainer.device)
    trainer.fit(model, data, train_loader, infer_loader, infer_layer=args.infer_layer)
    return trainer.best_acc


def test_all_experiments():
    import json
    file_path = "/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/optimize_pygs/res.json"
    status = json.load(open(file_path))

    for sampler in ['cluster', 'graphsage']: # mode
        for model in ['gcn', 'ggnn', 'gat', 'gaan']: # model
            for dataset in NEUROC_DATASET + SAINT_DATASET: # dataset
                if dataset == "com-amazon": continue
                for infer_layer in [True, False]:
                    try:
                        key = f'{sampler}_pyg15_{model}_{dataset}_{infer_layer}'
                        print("\n", key)
                        if key in status.keys() and status[key]: continue
                        experiment(model="pyg15_" + model, dataset=dataset, sampler=sampler, infer_layer=infer_layer, max_epoch=1)
                        status[key] = True
                    except Exception as e:
                        status[key] = False
                        print(e)
                        pass
    
    with open(file_path, 'w') as f:
        json.dump(status, f)

if __name__ == "__main__":
    # model, dataset, sampler, infer_layer = "gat", "pubmed", "graphsage", True
    # experiment(model="pyg15_" + model, dataset=dataset, sampler=sampler, infer_layer=infer_layer, max_epoch=1)
    test_all_experiments()
                    