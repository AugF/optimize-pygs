# 目标: box图
# memory随模型, batch_size, 数据集, mode的变化
import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from neuroc_pygs.options import get_args
from neuroc_pygs.configs import EXP_DATASET, ALL_MODELS, MODES, EXP_RELATIVE_BATCH_SIZE, PROJECT_PATH


def pics_figs(out_dir, args, keys):
    paras = {}
    for var0 in args[0]:
        paras[keys[0]] = var0
        for var1 in args[1]:
            paras[keys[1]] = var1
            for var2 in args[2]:
                paras[keys[2]] = var2
                paras[keys[3]] = ''
                file_name = '{}_{}_{}_{}'.format(paras['data'], paras['model'], paras['relative_batch_size'], paras['mode'])
                df_memory = {}
                df_edges = {}
                for var in args[3]:
                    paras[keys[3]] = var
                    in_file = '{}_{}_{}_{}'.format(paras['data'], paras['model'], paras['relative_batch_size'], paras['mode'])
                    real_path = osp.join(PROJECT_PATH, 'sec5_memory', 'batch_memory_info', in_file + '.csv')
                    print(real_path)
                    if not osp.exists(real_path):
                        break
                    df = pd.read_csv(real_path, index_col=0).to_dict(orient='list')
                    df_memory[var] = df['memory']
                    df_edges[var] = df['edges']
                if df_memory == {} or df_edges == {}:
                    continue
                fig, ax = plt.subplots(1, 2, figsize=(7, 5), tight_layout=True)
                ax[0].set_title('Memory')
                ax[0].boxplot(df_memory.values(), labels=df_memory.keys())
                ax[1].set_title('Edges')
                ax[1].boxplot(df_edges.values(), labels=df_edges.keys())
                fig.savefig(osp.join(PROJECT_PATH, 'sec5_memory', 'batch_memory_figs', out_dir, file_name + '.png'))
                plt.close()


if __name__ == '__main__':
    # pics_figs('modes', [EXP_DATASET, ALL_MODELS, EXP_RELATIVE_BATCH_SIZE, MODES], ['data', 'model', 'relative_batch_size', 'mode'])
    # pics_figs('relative_batch_sizes', [EXP_DATASET, ALL_MODELS, MODES, EXP_RELATIVE_BATCH_SIZE], ['data', 'model', 'mode', 'relative_batch_size'])
    # pics_figs('models', [EXP_DATASET, MODES, EXP_RELATIVE_BATCH_SIZE, ALL_MODELS], ['data', 'mode', 'relative_batch_size', 'model'])
    pics_figs('datasets', [ALL_MODELS, MODES, EXP_RELATIVE_BATCH_SIZE, EXP_DATASET], ['model', 'mode', 'relative_batch_size', 'data'])