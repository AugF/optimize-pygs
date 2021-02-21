import argparse
from optimize_pygs.datasets import build_dataset


parser = argparse.ArgumentParser()
# fmt: off
parser.add_argument('--dataset', '-dt', type=str, default="yelp",
                    help='Dataset')
# fmt: on
args = parser.parse_args()

graphsaint_data = ["ppi", "flickr", "reddit", "yelp", "amazon"]
neuroc_data = ['pubmed', 'amazon-photo', 'amazon-computers', 'coauthor-physics', 'com-amazon']

datasets = graphsaint_data + neuroc_data
for dataset_name in datasets:
    args.dataset = dataset_name
    data = build_dataset(args)
    print(dataset_name, data[0])