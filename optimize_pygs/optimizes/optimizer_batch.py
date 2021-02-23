import copy
import tqdm
import torch
import numpy as np

from optimize_pygs.options import get_default_args
from optimize_pygs.datasets import build_dataset
from optimize_pygs.models import build_model

from optimize_pygs.trainer import train_step, test_step

args = get_default_args(model="pyg15_gcn", dataset="cora")
print(args)

# 处理list
args.model, args.dataset = args.model[0], args.dataset[0]
dataset = build_dataset(args)
args.num_features = dataset.num_features
args.num_classes = dataset.num_classes
model = build_model(args)
model.set_loss_fn(dataset.get_loss_fn())
model.set_evaluator(dataset.get_evaluator())

print(dataset[0], model)
