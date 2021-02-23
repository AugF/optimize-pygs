from optimize_pygs.options import get_default_args
from optimize_pygs.datasets import build_dataset
from optimize_pygs.models import build_model
from optimize_pygs.loaders import build_sampler

args = get_default_args(model="pyg15_gcn", dataset="cora", sampler="cluster")

# 处理list
args.model, args.dataset = args.model[0], args.dataset[0]
print(args)

# step1. load dataset
dataset = build_dataset(args) # dataset_args

# step2. load model
args.num_features = dataset.num_features
args.num_classes = dataset.num_classes
model = build_model(args) # args
model.set_loss_fn(dataset.get_loss_fn())
model.set_evaluator(dataset.get_evaluator())

# step3. load sampler
args.sampler_data = dataset
train_loader = build_sampler(args)

num_batches = train_loader.get_num_batches()
train_loader.reset_iter()
for i in range(num_batches):
    batch = train_loader.get_next_batch()
    print(batch)
    