"""

"""
import torch

class BaseTrain:
    '''
    abstract class
    '''
    def __init__(self, sampler, model, evaluator, optimizer, loss, device):
        # device 信息
        self.sampler, self.model = sampler, model
        self.evaluator, self.optimizer = evaluator, optimizer
        self.loss, self.device = loss, device
    
    def _loss(self, y_pred, y_real):
        # 输出y_pred, y_real查看真实情况
        assert y_pred.shape[0] == y_real.shape[0]
        return self.loss(y_pred, y_real)
    
    def _pred(self, out):
        # 需要参考输出是什么维度的，这里可以统一假定为
        return 
    
    def train_step(self, node_subgraph, adj_subgraph):
        """
        Forward and backward propagation
        """
        self.model.train()
        self.optimizer.zero.grad()
        preds, labels, labels_converted = self.model(node_subgraph, adj_subgraph)
        loss = self._loss(preds, labels_converted) # labels.squeeze()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 5)
        self.optimizer.step()
        return loss, self.predict(preds), labels

    def eval_step(self, node_subgraph, adj_subgraph):
        """
        Forward propagation only
        """
        self.model.eval()
        with torch.no_grad():
            preds, labels, labels_converted = self.model(node_subgraph, adj_subgraph)
            loss = self._loss(preds, labels_converted)
        return loss, self.predict(preds), labels
    
    def train(self, x, loader):
        """
        后续提供用户自己实现
        """
        num = len(loader)
        total_loss = 0
        for batch in loader:
            loss, preds, labels = self.train_step(batch.x, batch.y)
        return total_loss            
    
    def test(self, x, loader):
        """
        后续可提供用户进行自己实现
        这里两种方式还需要进行细致的说明, 思考inference如何进行展开
        """
        num = len(loader)
        for batch in loader:
            loss, preds, labels = self.test_step(batch.x, batch.y)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', help="dataset: [cora, flickr, com-amazon, reddit, com-lj,"
                                                                    "amazon-computers, amazon-photo, coauthor-physics, pubmed]")
parser.add_argument('--model', type=str, default='gcn', help="gnn models: [gcn, ggnn, gat, gaan]")
parser.add_argument('--epochs', type=int, default=50, help="epochs for training")
parser.add_argument('--layers', type=int, default=2, help="layers for hidden layer")
parser.add_argument('--hidden_dims', type=int, default=64, help="hidden layer output dims")
parser.add_argument('--heads', type=int, default=8, help="gat or gaan model: heads")
parser.add_argument('--head_dims', type=int, default=8, help="gat model: head dims") # head_dims * heads = hidden_dims
parser.add_argument('--d_v', type=int, default=8, help="gaan model: vertex's dim") # d_v * heads = hidden_dims?
parser.add_argument('--d_a', type=int, default=8, help="gaan model: each vertex's dim in edge attention") # d_a = head_dims
parser.add_argument('--d_m', type=int, default=64, help="gaan model: gate: max aggregator's dim, default=64")

parser.add_argument('--x_sparse', action='store_true', default=False, help="whether to use data.x sparse version")
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--cpu', action='store_true', default=False, help='use cpu, not use gpu')
parser.add_argument('--device', type=str, default='cuda:1', help='[cpu, cuda:id]')
parser.add_argument('--lr', type=float, default=0.01, help="adam's learning rate")
parser.add_argument('--weight_decay', type=float, default=0.0005, help="adam's weight decay")
parser.add_argument('--no_record_shapes', action='store_false', default=True, help="nvtx or autograd's profile to record shape")
parser.add_argument('--json_path', type=str, default='', help="json file path for memory")

args = parser.parse_args()

