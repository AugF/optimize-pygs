import argparse
from optimize_pygs.datasets import build_dataset_from_name


def test_dataset(dataset_name):
    dataset = build_dataset_from_name(dataset_name)
    print(dataset_name, dataset.num_features, dataset.num_classes, dataset[0])


def test_all_datasets():
    EXP_DATASET = ['pubmed', 'amazon-photo', 'amazon-computers', 'ppi', 'flickr', 'reddit', 'yelp', 'amazon']
    for dataset_name in EXP_DATASET:
        test_dataset(dataset_name)


if __name__ == "__main__":
    # test_dataset("coauthor-physics")
    test_dataset()