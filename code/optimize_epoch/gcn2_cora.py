import os.path as osp

import re
import os
import time
import torch
# import torch.multiprocessing as mp
import multiprocessing as mp
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

# set
# os.environ['CUDA_VISIBLE_DEVICES']='0'

dataset = 'cora'
path = osp.join('/home/wangzhaokang/wangyunpan/gnns-project/datasets')
transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
dataset = Planetoid(path, dataset, transform=transform)
data = dataset[0]
data.adj_t = gcn_norm(data.adj_t)  # Pre-process GCN normalization.

# 暂时使用文件，后面考虑使用数据库
log_eval = "eval.log"

class Net(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super(Net, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset.num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels, dataset.num_classes))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x.log_softmax(dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5,
            shared_weights=True, dropout=0.6).to(device)
data = data.to(device)

optimizer = torch.optim.Adam([
    dict(params=model.convs.parameters(), weight_decay=0.01),
    dict(params=model.lins.parameters(), weight_decay=5e-4)
], lr=0.01)

print(next(model.parameters()).is_cuda)

def train(model, data, epoch):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    # with open(log_train, "a") as fw:
    #     fw.write(f'Epoch: {epoch:04d}, Loss: {loss:.4f}\n')
    return loss

def eval_callback(x):
    with open(log_eval, "a") as fw:
        fw.write(res)

@torch.no_grad()
def test(model, data, epoch):
    model.eval()
    pred, accs = model(data.x, data.adj_t).argmax(dim=-1), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    res = f'Epoch: {epoch:04d}, Train: {accs[0]:.4f}, Val: {accs[1]:.4f}, Test: {accs[2]:.4f}\n'
    with open(log_eval, "a") as fw:
        fw.write(res)
    return


st = time.time()
# 预热阶段，得到peck_memory和time
train_peak_memory, eval_peak_memory = [], []
for epoch in range(1, 10):
    loss = train(model, data, epoch)
    print(f'Epoch: {epoch:04d}, Loss: {loss:.4f}')
    train_peak_memory.append(torch.cuda.memory_stats(device)["allocated_bytes.all.peak"])
    torch.cuda.reset_peak_memory_stats(device) # 更新最大内存
    torch.cuda.empty_cache()
    test(model, data, epoch)
    train_peak_memory.append(torch.cuda.memory_stats(device)["allocated_bytes.all.peak"])

train_peak_memory = sum(train_peak_memory) / (1024 * 10240)
eval_peak_memory = sum(eval_peak_memory) / (1024 * 10240)
total_memory = 15079 # MB
print(f"peak memory, train: {train_peak_memory}MB, eval: {eval_peak_memory}MB")

cpu_eval = False
if train_peak_memory + eval_peak_memory >= total_memory:
    print("eval on cpu")
    model_eval = Net(hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5,
            shared_weights=True, dropout=0.6).to('cpu' if cpu_eval else device)
    data_eval = dataset[0].to('cpu' if cpu_eval else device)
    cpu_eval = True
else:
    model_eval, data_eval = model, data
    print("eval on gpu")

# print("test", id(data), id(data_eval), id(model), id(model_eval))
print(f"hot time: {time.time() - st}s")

mp.set_start_method('spawn', force=True)
t1 = time.time()
# 正式开始
processes = []

for epoch in range(10, 12):
    loss = train(model, data, epoch)
    print(f'Epoch: {epoch:04d}, Loss: {loss:.4f}')
    if epoch != 1000: # 这里通过时间来测试
        if cpu_eval:
            torch.save(model.state_dict(), 'tmp.pkl')
            model_eval.load_state_dict(torch.load('tmp.pkl',map_location=lambda storage, loc: storage))
        # if cpu_eval: # 异步
        if True:
            p = mp.Process(target=test, args=(model_eval, data_eval, epoch))
            p.start()
            processes.append(p)
        # else:
        #     p = torch.multiprocessing.Process(target=test, args=(model_eval, data_eval, epoch))
        #     p.start()
        #     processes.append(p)
    else:
        res = test(mode, data, epoch)
        with open(log_eval, "a") as fw:
            fw.write(res)     

# 收集结果
for p in processes:
    p.join()

t2 = time.time()

# 这里有不同的方式，还需要进行进一步尝试
best_val_acc = test_acc = 0
with open(log_eval, "r") as f:  
    for line in f:
        # 还需要查看是否是有效的
        match_line = re.match(r"Epoch: (.*), Train: (.*), Val: (.*), Test: (.*)", line)
        if match_line:
            epoch, train_acc, val_acc, tmp_test_acc = int(match_line.group(1)), float(match_line.group(2)), float(match_line.group(3)), float(match_line.group(4))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            print(f'Epoch: {epoch:04d}, Train: {train_acc:.4f}, '
                f'Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f}, '
                f'Final Test: {test_acc:.4f}')

ed = time.time()
print(f"train time: {t2 - t1}s")
print(f"handle data time: {ed - t2}s")
print(f"use time: {ed - st}s")