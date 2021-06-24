dataset_root = "/mnt/data/wangzhaokang/wangyunpan/mydata"

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

MODEL_PARAS = {
    'gcn': {'layers': [2, 3, 4, 5], 
            'hidden_dims': [16, 32, 64, 128, 256, 512, 1024, 2048]
            },
    'ggnn': {'layers': [2, 3, 4, 5],
             'hidden_dims': [16, 32, 64, 128, 256, 512, 1024, 2048]
            },
    'gat': {'layers': [2, 3, 4, 5],
             'head_dims': [16, 32, 64, 128, 256, 512, 1024, 2048],
             'heads': [1, 2, 4, 8, 16]
            },
    'gaan': {'layers': [2, 3, 4, 5],
             'gaan_hidden_dims': [16, 32, 64, 128, 256, 512, 1024, 2048],
             'heads': [1, 2, 4, 8, 16],
             'd_v': [8, 16, 32, 64, 128, 256],
             'd_a': [8, 16, 32, 64, 128, 256],
             'd_m': [8, 16, 32, 64, 128, 256]
            }
}


paras_df = {
    'gcn': ['nodes', 'edges', 'layers', 'features', 'classes', 'hidden_dims', 'memory'],
    'gat': ['nodes', 'edges', 'layers', 'features', 'classes', 'heads', 'head_dims', 'memory'],  
}