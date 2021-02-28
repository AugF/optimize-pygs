from collections import defaultdict

res = defaultdict(list)
x_train = defaultdict(list)
res['a'].append(1)
x_train['a'].append(1)
x_train.update(res)
# for k, v in res.items(): # update x_train
#     print(x_train[k])
#     x_train[k].extend(v)
print(x_train)