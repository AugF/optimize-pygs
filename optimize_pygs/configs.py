DEFAULT_MODEL_CONFIGS = {
    'pyg_gcn': {
        "num_features": 100,
        "num_classes": 12,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.5
    },
    'pyg_gat': {
        "num_features": 100,
        "num_classes": 12,
        "hidden_size": 64,
        "num_heads": 2,
        "dropout": 0.5
    },
    'pyg_sage': {
        "num_features": 100,
        "num_classes": 12,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.5
    },
    'pyg15_gcn': {
        "num_features": 100,
        "num_classes": 12,
        "hidden_size": 64,
        "num_layers": 2,
        "norm": None,
    },
    'pyg15_ggnn': {
        "num_features": 100,
        "num_classes": 12,
        "hidden_size": 64,
        "num_layers": 2,
    },
    'pyg15_gat': {
        "num_features": 100,
        "num_classes": 12,
        "head_size": 8,
        "num_layers": 2,
        "heads": 8,
    },
    'pyg15_gaan': {
        "num_features": 100,
        "num_classes": 12,
        "hidden_size": 8,
        "num_layers": 2,
        "heads": 8,
        "d_v": 8,
        "d_a": 8,
        "d_m": 8
    }
}


# node classification task
BEST_CONFIGS = {
    'pyg_gcn': {
        'general': {'lr'},
        'cora': {}
    }
}