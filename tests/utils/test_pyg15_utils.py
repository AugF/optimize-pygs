from optimize_pygs.utils import get_datasets

def test_get_datasets():
    neuroc_data = ['pubmed', 'amazon-photo', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    for name in neuroc_data:
        dataset = get_datasets(name)
        data = dataset[0]
        print(name, data)


if __name__ == "__main__":
    test_get_datasets()
