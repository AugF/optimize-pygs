from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from code.globals import *
from code.utils.tools import log_dir, parse_n_prepare

"""
TODO: 封装为pipeline, 即最后写成抽象类
"""
def train(model, train_loader, x, y, optimizer, device):
    model.train()
    
    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)

    pbar.close()

    train_loss = total_loss / len(train_loader)
    train_acc = total_correct / train_idx.size(0)
    
    return train_loss, train_acc

@torch.no_grad()
def test(mode, subgraph_loader, x, y, evaluator, split_idx, device):
    model.eval()

    out = model.inference(x, subgraph_loader, device)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, val_acc, test_acc


def prepare_data(data_prefix, root="/home/wangzhaokang/wangyunpan/gnns-project/datasets"):
    dataset = PygNodePropPredDataset(date_prefix, root=root)
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name=data_prefix)
    data = dataset[0]
    x, y = data.x, data.y.squeeze()
    return x, y, evaluator, split_idx


def prepare_model(**args):
    return SAGE(args['num_features'], args['hidden_dims'], args['num_classes'], args['num_layers'])


if __name__ == "__main__":
    # set random seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if args_global.gpu >=1 and torch.cuda.is_availabel():
        torch.cuda.manual_seed(SEED)
        device = torch.device(f"cuda:{args_global.gpu}")
    else:
        device = torch.device("cpu")

    # step1: 创建log目录, 获取相关参数
    log_dir = log_dir(args_global.train_config, args_global.data_prefix, git_branch, git_rev, timestamp)
    model_paras, train_paras, phase_paras = parse_n_prepare(args_global.train_config)

    if 'eval_val_every' not in train_params:
        train_paras['eval_val_every'] = EVAL_VAL_EVERY_EP    
        
    # step2: 准备数据集
    x, y, evaluator, split_idx = prepare_data(args_global.data_prefix)

    # step3: model, optimizer
    model = prepare_model(num_features=x.shape[1], num_classes=y.shape[1], hidden_dims=model_paras['dim'], num_layers=model_paras['layer'])
    
    # 运行
    logger = Logger(args_global.runs, args_global)
    for run in range(args_global.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters, lr=train_paras['lr'], weight_decay=train_paras['weight_decay'])
        for epoch in range(args_global.epochs):
            train(model, train_loader, x, y, optimizer, devcice)
            if (epoch + 1) % train_paras['eval_val_every'] == 0:
                if args_global.cpu_eval:
                    torch.save(model.state_dict(), log_dir + "/tmp.pkl")
                    model_eval.load_state_dict
                
        logger.print_statistics(run)
    logger.print_statistics()

    
