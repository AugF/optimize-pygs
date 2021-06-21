import sys
import argparse
import subprocess

default_args = ' '.join(sys.argv[1:]
    
# 训练与评估步骤的流水线(--device)
pro_train = subprocess.Popen(f'python train.py ' + default_args, shell=True)
pro_eval = subprocess.Popen(f'python eval.py ' + default_args, shell=True)

pro_train.communicate()
pro_train.wait()

pro_eval.communicate()
pro_eval.wait()
pro_train.kill()

