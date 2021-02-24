import argparse
from optimize_pygs.datasets import build_dataset_from_name


def test_dataset(dataset_name):
    data = build_dataset_from_name(dataset_name)
    print(dataset_name, data[0])


def test_all_datasets():
    graphsaint_data = ["ppi", "flickr", "reddit", "yelp", "amazon"]
    neuroc_data = ['pubmed', 'amazon-photo', 'amazon-computers', 'coauthor-physics', 'com-amazon']

    datasets = graphsaint_data + neuroc_data
    for dataset_name in datasets:
        test_dataset(dataset_name)


if __name__ == "__main__":
    # test_dataset("coauthor-physics")
    test_all_datasets()