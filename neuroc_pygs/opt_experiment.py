import sys
import argparse
import subprocess

default_args = ' '.join(sys.argv[1:])

if '--mode None' in default_args:
    pro_eval = subprocess.Popen(f'python eval_full.py --device cuda:0 ' + default_args, shell=True)
else:
    pro_eval = subprocess.Popen(f'python eval_sampling.py --device cuda:1 ' + default_args, shell=True)
    
# 训练与评估步骤的流水线(--device)
pro_train = subprocess.Popen(f'python train.py --device cuda:0 ' + default_args, shell=True)

pro_eval.communicate()
pro_eval.wait()
pro_train.kill()

