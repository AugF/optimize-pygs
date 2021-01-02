import os, sys, time, datetime
import torch
import numpy as np
import argparse
import subprocess

import subprocess
git_rev = subprocess.Popen("git rev-parse --short HEAD", shell=True, stdout=subprocess.PIPE, universal_newlines=True).communicate()[0]
git_branch = subprocess.Popen("git symbolic-ref --short -q HEAD", shell=True, stdout=subprocess.PIPE, universal_newlines=True).communicate()[0]

timestamp = time.time()
timestamp = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H-%M-%S')

# 参数配置
parser = argparse.ArgumentParser(description="argument for optimize-pygs")
parser.add_argument("--seed", default=123, type=int, help="random seed for training")
parser.add_argument("--runs", default=2, type=int, help="total runs for training")
parser.add_argument("--epochs", default=50, type=int, help="total epochs for training")
parser.add_argument("--eval_train_every",default=15,type=int,help="how often to evaluate training subgraph accuracy")

parser.add_argument("--model", default="SAGE", type=str, help="models for training")
parser.add_argument("--train_config", default="", type=str, help="path to the configuration of training (*.yml)")
parser.add_argument("--data_prefix", default="ogbn-products", type=str, help="dataset for training")

parser.add_argument("--gpu",default="-1234",type=str,help="which GPU to use")
parser.add_argument("--num_cpu_core", default=20, type=int, help="Number of CPU cores for training")
# TODO: 是否需要增加内存资源的判定
parser.add_argument("--dir_log",default=".",type=str,help="base directory for logging and saving embeddings")
parser.add_argument("--saved_model_path",default="",type=str,help="path to pretrained model file")
args_global = parser.parse_args()

NUM_CPU_CORE = args_global.num_cpu_core
EVAL_VAL_EVERY_EP = 1       # get accuracy on the validation set every this # epochs
SEED = args_global.seed

# auto choosing available NVIDIA GPU
gpu_selected = args_global.gpu
if gpu_selected == '-1234':
    # auto detect gpu by filtering on the nvidia-smi command output
    gpu_stat = subprocess.Popen("nvidia-smi",shell=True,stdout=subprocess.PIPE,universal_newlines=True).communicate()[0]
    gpu_avail = set([str(i) for i in range(8)])
    for line in gpu_stat.split('\n'):
        if 'python' in line:
            if line.split()[1] in gpu_avail:
                gpu_avail.remove(line.split()[1])
            if len(gpu_avail) == 0:
                gpu_selected = -2
            else:
                gpu_selected = sorted(list(gpu_avail))[0]
    if gpu_selected == -1:
        gpu_selected = '0'
    args_global.gpu = int(gpu_selected)
if str(gpu_selected).startswith('nvlink'):
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_selected).split('nvlink')[1]
elif int(gpu_selected) >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_selected)
    GPU_MEM_FRACTION = 0.8
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
args_global.gpu = int(args_global.gpu)
