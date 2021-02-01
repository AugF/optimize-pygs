import copy
import torch.nn.functional as F
import torch.multiprocessing as mp
from code.optimize_batch.tasks import model, loader, optimizer, device

def task1(q1, loader, num):
    loader_iter = iter(loader)
    for i in range(num):
        data = next(loader_iter)
        q1.put(copy.copy(data.pin_memory()))
        
def task3(q2, model, optimizer, device, num):
    total_loss = total_examples = total_correct = 0
    for i in range(num):
        data = q2.get().to(device)
        # if i == 0:
        #     print(f"task3 begin: {time.time() - st1}s")
        if data.train_mask.sum() == 0: # task3
            continue
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)[data.train_mask]
        y = data.y.squeeze(1)[data.train_mask]
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()

        num_examples = data.train_mask.sum().item()
        total_loss += loss.item() * num_examples
        total_examples += num_examples

        total_correct += out.argmax(dim=-1).eq(y).sum().item()
        total_examples += y.size(0)
    print(f'loss: {total_loss / total_examples:.4f}, train_acc: {total_correct / total_examples:.4f}')
    return


if __name__ == "__main__":
    model.train()
    num = len(loader)
    ctx = mp.get_context('spawn')
    q1, q2 = ctx.Queue(maxsize=1), ctx.Queue(maxsize=1)
    job1 = ctx.Process(target=task1, args=(q1, loader, num))
    # job2 = mp.Process(target=task2, args=(q1, q2, ))
    job3 = ctx.Process(target=task3, args=(q1, model, optimizer, device, num))
    
    job1.start()
    # job2.start()
    job3.start()
    
    job1.join()
    # job2.join()
    job3.join()
