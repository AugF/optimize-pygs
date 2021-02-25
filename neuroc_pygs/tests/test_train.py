import os
import sys
import time

from neuroc_pygs.configs import ALL_MODELS, EXP_DATASET, MODES


path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
cmd = 'python {}/train.py --mode {} --model {} --dataset {} --epochs 1'
for data in EXP_DATASET:
    for mode in MODES:
        for model in ALL_MODELS:
            try:
                real_cmd = cmd.format(path, mode, model, data)
                print(real_cmd)
                os.system(real_cmd)
            except Exception as e:
                print(e)
