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
    'snap_1000': 'accuracy',
    'snap_10000': 'accuracy',
    'snap_100000': 'accuracy',
    'snap_1000000': 'accuracy'
}

ALL_MODELS = ['gcn', 'gat', 'ggnn', 'gaan']
MODES = ['cluster', 'graphsage']
PROJECT_PATH = "/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs"
EXP_DATASET = ['amazon-photo', 'amazon-computers', 'ppi', 'pubmed', 'flickr', 'reddit', 'yelp', 'amazon']
EXP_RELATIVE_BATCH_SIZE = [0.01, 0.03, 0.06, 0.1, 0.25, 0.5]
