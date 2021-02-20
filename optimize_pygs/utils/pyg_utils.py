import re
import os.path as osp 


def to_list(x):
    if not isinstance(x, (tuple, list)):
        x = [x]
    return x


def files_exist(files):
    return len(files) != 0 and all(osp.exists(f) for f in files)


def __repr__(obj):
    if obj is None:
        return 'None'
    return re.sub('(<.*?)\\s.*(>)', r'\1\2', obj.__repr__())