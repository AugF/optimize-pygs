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