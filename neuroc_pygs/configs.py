dataset_root = "/mnt/data/wangzhaokang/wangyunpan/data"


DATASET_METRIC = {
    'cora': 'accuracy',
    'pubmed': 'accuracy',
    'amazon-photo': 'accuracy',
    'amazon-computers': 'accuracy',
    'coauthor-physics': 'accuracy',
    'com-amazon': 'accuracy',
    'ppi': 'multilabel_f1',
    'flickr': 'accuracy',
    'reddit': 'accuracy',
    'yelp': 'multilabel_f1',
    'amazon': 'multilabel_f1',
}

ALL_MODELS = ['gcn', 'gat', 'ggnn', 'gaan']
MODES = ['cluster', 'graphsage']
PROJECT_PATH = "/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs"
EXP_DATASET = ['pubmed', 'amazon-photo', 'amazon-computers', 'ppi', 'flickr', 'reddit', 'yelp', 'amazon']
