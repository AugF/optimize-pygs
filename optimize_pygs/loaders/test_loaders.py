from optimize_pygs.options import get_default_args
from optimize_pygs.datasets import build_dataset
from optimize_pygs.models import build_model
from optimize_pygs.loaders import build_sampler_from_name
from optimize_pygs.loaders.configs import TRAIN_CONFIG, INFER_CONFIG

args = get_default_args(model="pyg15_gcn", dataset="flickr", sampler="graphsage")
print(args)
# step1. load dataset
dataset = build_dataset(args) # dataset_args

# step2. load model
args.num_features = dataset.num_features
args.num_classes = dataset.num_classes
model = build_model(args) # args
model.set_loss_fn(dataset.get_loss_fn())
model.set_evaluator(dataset.get_evaluator())

print(args)
train_loader = build_sampler_from_name(args.sampler, dataset=dataset, 
                num_parts=args.num_parts, batch_size=args.batch_size, num_workers=args.num_workers, 
                **TRAIN_CONFIG[args.sampler])
subgraph_loader = build_sampler_from_name(args.infer_sampler, dataset=dataset,
                batch_size=args.infer_batch_size, num_workers=args.num_workers, 
                **INFER_CONFIG[args.infer_sampler])

# num_batches = train_loader.get_num_batches()
# train_loader.reset_iter()
# for i in range(num_batches):
#     batch = train_loader.get_next_batch()
#     print(batch)

# num_batches = subgraph_loader.get_num_batches()
# subgraph_loader.reset_iter()
# for i in range(num_batches):
#     batch = subgraph_loader.get_next_batch()
#     print(batch)