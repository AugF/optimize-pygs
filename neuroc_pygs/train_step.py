import torch
import numpy as np


def train(model, data, train_loader, optimizer, mode, device):
    model.train()
    all_loss = []
    loader_iter, loader_num = iter(train_loader), len(train_loader)
    for i in range(loader_num):
        optimizer.zero_grad()
        if mode == "cluster":
            # task1
            batch = next(loader_iter)
            # task2
            batch = batch.to(device)
            # task3
            logits = model(batch.x, batch.edge_index)
            loss = model.loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
        else:
            # task1
            batch_size, n_id, adjs = next(loader_iter)
            x, y = data.x[n_id], data.y[n_id[:batch_size]]
            # task2
            x, y = x.to(device), y.to(device)
            adjs = [adj.to(device) for adj in adjs]
            # task3
            logits = model(x, adjs)
            loss = model.loss_fn(logits, y)
            loss.backward()
        optimizer.step()
        all_loss.append(loss.item())
    return np.mean(all_loss)


@torch.no_grad()
def test(model, data, subgraph_loader, device, split="val"):
    model.eval()
    all_loss, all_acc = [], []
    loader_iter, loader_num = iter(subgraph_loader), len(subgraph_loader)
    for i in range(loader_num):
        # start 
        batch_size, n_id, adjs = next(loader_iter)
        x, y = data.x[n_id], data.y[n_id[:batch_size]]
        # task2
        x, y = x.to(device), y.to(device)
        adjs = [adj.to(device) for adj in adjs]
        # task3
        logits = model(x, adjs)
        loss = model.loss_fn(logits, y)
        loss.backward()
        acc = model.evaluator(logits, y) / batch_size
        # end
        all_loss.append(loss.item())
        all_acc.append(acc)
    return np.mean(all_acc), np.mean(all_loss)


@torch.no_grad()
def infer(model, data, subgraph_loader, device, split='val'):
    model.eval()
    y_pred = model.inference(data.x, subgraph_loader)
    y_true = data.y.cpu()

    mask = getattr(data, split + "_mask")
    loss = model.loss_fn(y_pred[mask], y_true[mask])
    acc = model.evaluator(y_pred[mask], y_true[mask]) 
    return acc, loss
