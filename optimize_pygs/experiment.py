import torch
import numpy as np
from optimize_pygs.options import get_default_args
from optimize_pygs.datasets import build_dataset
from optimize_pygs.models import build_model
from optimize_pygs.loaders import build_sampler_from_name
from optimize_pygs.loaders.configs import TRAIN_CONFIG, INFER_CONFIG
from optimize_pygs.trainers import build_trainer


def experiment(model, dataset, sampler, infer_layer=False, **kwargs):
    args = get_default_args(
        model="pyg15_gcn", dataset="flickr", sampler="cluster", **kwargs)
    # step1. load dataset
    dataset = build_dataset(args)  # dataset_args
    data = dataset[0]

    # step2. load model
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    model = build_model(args)  # args
    model.set_loss_fn(dataset.get_loss_fn())
    model.set_evaluator(dataset.get_evaluator())
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.manual_seed(args.seed)
    
    # step3 load trainer
    trainer = build_trainer(args)

    # step4 training
    model, data = model.to(trainer.device), data.to(trainer.device)

    train_loader = build_sampler_from_name(args.sampler, dataset=dataset,
                                           num_parts=args.num_parts, batch_size=args.batch_size, num_workers=args.num_workers,
                                           **TRAIN_CONFIG[args.sampler])

    if args.infer_layer:
        infer_loader = build_sampler_from_name(args.infer_sampler, dataset=dataset, sizes=[-1],
                                           batch_size=args.infer_batch_size, num_workers=args.num_workers,
                                           **INFER_CONFIG[args.infer_sampler])
    else:
        infer_loader = build_sampler_from_name(args.infer_sampler, dataset=dataset, sizes=[-1] * args.num_layers,
                                              batch_size=args.infer_batch_size, num_workers=args.num_workers,
                                              **INFER_CONFIG[args.infer_sampler])
        
    trainer.fit(model, data, train_loader, infer_loader, infer_layer=args.infer_layer)
    return trainer.best_acc


if __name__ == "__main__":
    from optimize_pygs.configs import NEUROC_DATASET, SAINT_DATASET
    for sampler in ['cluster', 'graphsage']: # mode
        for model in ['gcn', 'ggnn', 'gat', 'gaan']: # model
            for dataset in NEUROC_DATASET + SAINT_DATASET: # dataset
                for infer_layer in [True, False]:
                    try:
                        print("\n", sampler, "pyg15_" + model, dataset, infer_layer)
                        experiment(model="pyg15_" + model, dataset=dataset, sampler=sampler, infer_layer=infer_layer, max_epoch=2)
                    except Exception as e:
                        print(e)
                        pass
                    