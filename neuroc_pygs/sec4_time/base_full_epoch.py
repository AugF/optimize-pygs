import math
import time
import numpy as np
from neuroc_pygs.options import get_args, build_dataset, build_model_optimizer
from neuroc_pygs.sec4_time.epoch_utils import train_full, test_full


def epoch(): # 50取平均，删除一些异常元素
    args = get_args()
    print(args)
    data = build_dataset(args)
    model, optimizer = build_model_optimizer(args, data)
    model, data = model.to(args.device), data.to(args.device)
    for epoch in range(args.epochs): # 50
        st_time = time.time()
        train_full(model, data, optimizer)
        accs = test_full(model, data)
        ed_time = time.time()
        print(f'Epoch: {epoch:03d}, st_time: {st_time}, ed_time: {ed_time}, use_time: {ed_time - st_time}')
    return


if __name__ == '__main__':
    epoch()