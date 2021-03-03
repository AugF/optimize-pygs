# 测试inference, evaluation, train三个阶段
# 通过添加了opt_loader来表示是否开启该优化
# 
import time
from neuroc_pygs.options import build_train_loader, get_args, build_dataset

args = get_args()

data = build_dataset(args)
train_loader = build_train_loader(args, data)

iter1 = iter(train_loader)
iter2 = BackgroundGenerator(iter(train_loader))
t1 = time.time()
for _ in iter1: pass
t2 = time.time()
for _ in iter2: pass
t3 = time.time()
print(f'use time: {t2 - t1}, opt time: {t3 - t2}')