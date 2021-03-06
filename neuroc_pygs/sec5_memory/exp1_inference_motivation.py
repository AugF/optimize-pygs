


@torch.no_grad()
def test(model, data, subgraph_loader, args, df, cnt):
    model.eval()
    device = args.device
    loader_iter, loader_num = iter(subgraph_loader), len(subgraph_loader)
    if opt_loader:
        loader_iter = BackgroundGenerator(loader_iter)
    for i in range(loader_num):
        batch_size, n_id, adjs = next(loader_iter)
        x, y = data.x[n_id], data.y[n_id[:batch_size]]
        x, y = x.to(device), y.to(device)
        adjs = [adj.to(device) for adj in adjs]

        df['nodes'].append(adjs[0][2][0])
        df['edges'].append(adjs[0][0].shape[1])
        logits = model(x, adjs)
        loss = model.loss_fn(logits, y)
        acc = model.evaluator(logits, y) / batch_size

        df['memory'].append(torch.cuda.memory_stats(device)["allocated_bytes.all.peak"])
        torch.cuda.reset_max_memory_allocated(device)
        cnt += 1
        if cnt >= 40:
            break        
    return df, cnt


@torch.no_grad()
def infer(model, data, subgraph_loader, args, opt_loader=False):
    device, log_batch, log_batch_dir = args.device, args.log_batch, args.log_batch_dir
    model.eval()
    y_pred = model.inference(data.x, subgraph_loader, log_batch, opt_loader)
    y_true = data.y.cpu()

    mask = getattr(data, split + "_mask")
    loss = model.loss_fn(y_pred[mask], y_true[mask])
    acc = model.evaluator(y_pred[mask], y_true[mask]) 
    return acc, loss


from neuroc_pygs.options import get_args, build_dataset, build_subgraphloader, build_model
args.infer_layer = True
args = get_args()
data = build_dataset(args)
model = build_model(args, data)
subgraphloader = build_subgraphloader(args, data)

model = model.to(args.device)
num_loader = len(subgraphloader)
for i in range(40):
    if num_loader * i >= 40:
        break
    infer(model, data, subgraph_loader, args)