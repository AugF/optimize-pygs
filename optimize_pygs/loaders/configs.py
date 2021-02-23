TRAIN_CONFIG = {
    'cluster': {
        # 'num_parts': 1500,
        'recursive': True,
        # 'batch_size': 20,
        'shuffle': True,
        # 'num_workers': 12
    },
    'graphsage': {
        'node_idx': None,
        'sizes': [25, 10], 
        # 'batch_size': 1024,
        'shuffle': True,
        # 'num_workers': 12
    }
}

INFER_CONFIG = {
    'graphsage': {
        'node_idx': None,
        # 'sizes': [-1, -1], 
        # 'batch_size': 1024,
        'shuffle': False,
        # 'num_workers': 12
    }
}