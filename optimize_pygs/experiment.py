"""
import copy
import tqdm
import torch
import numpy as np

from optimize_pygs.options import get_default_args
from optimize_pygs.datasets import build_dataset
from optimize_pygs.models import build_model

from optimize_pygs.trainer import train_step, test_step
from optimize_pygs.loaders import build_loader
from optimize_pygs.trainers.early_stopping import NoStoppingCriterion

args = get_default_args(model="pyg15_gcn", dataset="cora")
print(args)

# 处理list
args.model, args.dataset = args.model[0], args.dataset[0]


dataset = build_dataset(args)
train_loader, subgraph_loader = build_loader(args) # loader, mini-batch

# model相关
args.num_features = dataset.num_features
args.num_classes = dataset.num_classes
model = build_model(args)
model.set_loss_fn(dataset.loss_fn)
model.set_evaluator(dataset.evaluator)

print(dataset[0], model)

def fit(): # get best_model
    epoch_iter = tqdm(range(args.max_epoch))
    patience = 0
    max_score = 0
    min_loss = np.inf
    best_model = None 

    early_stopping = NoStoppingCriterion() # early_stopping
    early_stopping.reset()

    for epoch in epoch_iter:
        train_acc, _ = train_step(model, dataset, train_loader)
        val_acc, val_loss = test_step(model, dataset, subgraph_loader, split="val")
        
        epoch_iter.set_description(f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}")
        if early_stopping.should_stop(epoch, val_acc, val_loss):
            print("early_stopping ...")
            break

    early_stopping.after_stopping_ops()
    return torch.load_dict(early_stopping.best_model)


# final test, transductive
def predict(model, data):
    acc, _ = test_step(model, data, subgraph_loader, val="test") #?test_loader
    return acc
"""