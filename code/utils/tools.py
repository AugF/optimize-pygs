import numpy as np

import torch
from torch.autograd import Variable
from code.globals import *
import yaml
from yaml import Loader

def to_numpy(x):
    if isinstance(x, Variable):
        x = x.data
    return x.cpu().numpy() if x.is_cuda else x.numpy()

def log_dir(f_train_config, prefix, git_branch, git_rev,timestamp):
    import getpass
    log_dir = args_global.dir_log + "/log_train/" + prefix.split("/")[-1]
    log_dir += "/{ts}-{model}-{gitrev:s}/".format(
            model=args_global.model,
            gitrev=git_rev.strip(),
            ts=timestamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if f_train_config != '':
        from shutil import copyfile
        copyfile(f_train_config,'{}/{}'.format(log_dir,f_train_config.split('/')[-1]))
    return log_dir


def parse_n_prepare(f_train_config):
    with open(f_train_config) as f_train_config:
        train_config = yaml.load(f_train_config, Loader=Loader)
    model_paras = {
        'dim': -1,
        'layer': 2, 
        'aggr': 'concat',
        'loss': 'softmax',
        'act': 'relu',
        'bias': 'norm',
        'attention': 32,
        'heads': 4
    }
    model_paras.update(train_config['network'][0])
    train_paras = {
        'lr': 0.01,
        'weight_decay': '0.006',
        'dropout': 0.5,
        'attention_dropout': 0.3,
        'eval_val_every': 1
    }
    train_paras.update(train_config['paras'][0])
    phase_paras = {
        'sampler': 'neighbor_sampler'
    }
    phase_paras.update(train_config['phase'][0])
    return model_paras, train_paras, phase_paras
    