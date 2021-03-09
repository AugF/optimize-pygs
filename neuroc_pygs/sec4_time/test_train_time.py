def train(model, optimizer, data, loader, device, mode, df, non_blocking=False):
    model.reset_parameters()
    for v in model.parameters():
        print(v)
    model.train()
    all_loss = []
    loader_num, loader_iter = len(loader), iter(loader)
    for _ in range(loader_num):
        t0 = time.time()
        if mode == 'cluster':
            batch = next(loader_iter)
            t1 = time.time()
            batch = batch.to(device, non_blocking=non_blocking)
            t2 = time.time()
            batch_size = batch.train_mask.sum().item()
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index)
            loss = model.loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
        elif mode == 'graphsage':
            batch_size, n_id, adjs = next(loader_iter)
            x, y = data.x[n_id], data.y[n_id[:batch_size]]
            t1 = time.time()
            x, y = x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)
            adjs = [adj.to(device, non_blocking=non_blocking) for adj in adjs]
            t2 = time.time()
            # task3
            print(x.shape, adjs[0][2])
            optimizer.zero_grad()
            logits = model(x, adjs)
            loss = model.loss_fn(logits, y)
            loss.backward()
            optimizer.step()
        all_loss.append(loss.item() * batch_size)
        t3 = time.time()
        print('use time', t3 - t2)
        df.append([t1 - t0, t2 - t1, t3 - t2])
        # print(f'Batch {_}: sampling time: {t1-t0}, to_time: {t2-t1}, training_time: {t3-t2}')
    return np.sum(all_loss) / int(data.train_mask.sum())

if __name__ == '__main__':
    import time, sys
    import numpy as np
    from neuroc_pygs.options import build_train_loader, get_args, build_dataset, build_model_optimizer
    # from neuroc_pygs.sec4_time.exp2_sampling_training import train
    sys.argv = [sys.argv[0], '--device', 'cuda:1']
    args = get_args()
    args.mode = 'graphsage'
    data = build_dataset(args)
    model, optimizer = build_model_optimizer(args, data)
    model = model.to(args.device)
    train_loader = build_train_loader(args, data)

    # for loader in train_loader:
    #     batch_size, n_id, adjs = loader
    #     print(n_id)
    res = []
    train(model, optimizer, data, train_loader, args.device, args.mode, res)
    res = np.array(res)
    print(np.mean(res, axis=0))