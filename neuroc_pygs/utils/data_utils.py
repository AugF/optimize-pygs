import torch
import json


def single_apply(data, item, func):
    if torch.is_tensor(item):
        return func(item)
    elif isinstance(item, (tuple, list)):
        return [(data, v, func) for v in item]
    elif isinstance(item, dict):
        return {k: apply(data, v, func) for k, v in item.items()}
    else:
        return item

def apply(data, func, *keys):
    r"""Applies the function :obj:`func` to all tensor attributes
    :obj:`*keys`. If :obj:`*keys` is not given, :obj:`func` is applied to
    all present attributes.
    """
    for key, item in data(*keys):
        data[key] = data.__apply__(item, func)
    return data


def to(data, device, *keys, **kwargs):
    return apply(data, lambda x: x.cuda(device, **kwargs), *keys) 


def print_device(data):
    for key, value in data.__dict__.items():
        if hasattr(value, 'device'):
            print(key, value.device)


class BatchLogger:
    def __init__(self, name='Log Batch'):
        self.name = name
        self.cnt = 0
        self.sample_time, self.to_time, self.train_time = [], [], []
        self.avg_sample_time, self.avg_to_time, self.avg_train_time = 0, 0, 0
    
    def add_batch(self, sample_time, to_time, train_time):
        self.sample_time.append(sample_time)
        self.to_time.append(to_time)
        self.train_time.append(train_time)
        self.cnt += 1

    def print_batch(self):
        for i in range(self.cnt):
            print(f"Batch {i}, sample_time: {self.sample_time[i]:.8f}, "
                    f"to_time: {self.to_time[i]:.8f}, train_time: {self.train_time[i]:.8f}")
        self.avg_sample_time = sum(self.sample_time) / self.cnt
        self.avg_to_time = sum(self.to_time) / self.cnt
        self.avg_train_time = sum(self.train_time) / self.cnt
        print(f"Avg: sample_time: {self.avg_sample_time:.8f}, "
                    f"to_time: {self.avg_to_time:.8f}, "
                    f"train_time: {self.avg_train_time:.8f}")

    def reset(self):
        self.sample_time, self.to_time, self.train_time = [], [], []
        self.cnt = 0
    
    def save(self, path):
        df = {}
        for k, v in self.__dict__.items():
            df[k] = v
        with open(path, 'w') as f:
            json.dump(df, f)

    def load(self, path):
        all_time = json.load(open(path))
        for k, v in all_time.items():
            self.__setattr__(k, v)


class EpochLogger: # Epoch, 时间只需要均摊到每个Epoch即可
    def __init__(self, name='Log Batch'):
        self.name = name
        self.cnt = 0
        self.train_time, self.eval_time = [], []
        self.avg_train_time, self.avg_eval_time = 0, 0
    
    def add_epoch(self, train_time, eval_time):
        self.train_time.append(train_time)
        self.eval_time.append(eval_time)
        self.cnt += 1

    def print_epoch(self):
        for i in range(self.cnt):
            print(f"Epoch {i}, train_time: {self.train_time[i]:.8f}, "
                    "train_time: {self.eval_time[i]:.8f}")
        self.avg_train_time = sum(self.train_time) / self.cnt
        self.avg_eval_time = sum(self.eval_time) / self.cnt
        print(f"Avg: sample_time: {self.avg_sample_time:.8f}, "
                    "train_time: {self.avg_train_time:.8f}")

    def reset(self):
        self.train_time, self.eval_time = [], [], []
        self.cnt = 0
    
    def save(self, path):
        df = {}
        for k, v in self.__dict__.items():
            df[k] = v
        with open(path, 'w') as f:
            json.dump(df, f)

    def load(self, path):
        all_time = json.load(open(path))
        for k, v in all_time.items():
            self.__setattr__(k, v)


if __name__ == "__main__":
    logger = BatchLogger()
    print(logger.__dict__)