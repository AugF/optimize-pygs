import torch

import torch.nn.functional as F
from torch.nn import Parameter, Module

from optimize_pygs.layers import GatedGraphConv
from optimize_pygs.utils.inits import glorot, zeros
from optimize_pygs.utils.pyg15_utils import nvtx_push, nvtx_pop, log_memory
from . import BaseModel, register_model


@register_model("pyg15_ggnn")
class GGNN(BaseModel):
    """
    GGNN layer
    """
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser.
        """
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--num-layers", type=int, default=2)
        # fmt: on
        
    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.num_layers,
            device = "cpu" if not torch.cuda.is_available() or args.cpu else args.device_id,
            gpu=args.nvtx_flag,
            flag=args.memory_flag,
            infer_flag=args.infer_flag
        )
        
    def __init__(self, num_features, hidden_size, num_classes, num_layers, gpu=False, device="cpu", flag=False, infer_flag=False):
        super(GGNN, self).__init__()
        self.num_features, self.num_classes = num_features, num_classes
        self.num_layers, self.hidden_size = num_layers, hidden_size
        self.gpu = gpu
        self.device = device
        self.flag, self.infer_flag = flag, infer_flag

        self.weight_in = Parameter(torch.Tensor(num_features, hidden_size))
        self.weight_out = Parameter(torch.Tensor(hidden_size, num_classes))
        self.convs = torch.nn.ModuleList([GatedGraphConv(out_channels=hidden_size, num_layers=num_layers, gpu=gpu,
                                                         flag=flag, infer_flag=infer_flag, device=device)])
        glorot(self.weight_in)
        glorot(self.weight_out)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, x, adjs):
        device = torch.device(self.device)
        nvtx_push(self.gpu, "input-transform")
        x = torch.matmul(x, self.weight_in)
        
        nvtx_pop(self.gpu)
        log_memory(self.flag, device, "input-transform")

        x = self.convs[0](x, adjs)
        nvtx_push(self.gpu, "output-transform")
        x = torch.matmul(x, self.weight_out)
        nvtx_pop(self.gpu)
        log_memory(self.flag, device, "output-transform")
        return x

    def inference(self, x_all, subgraph_loader):
        device = torch.device(self.device)

        x_all = torch.matmul(x_all.to(device), self.weight_in) # 尽最大可能第键槽内存
        log_memory(self.infer_flag, device, "input-transform")
        
        x_all = self.convs[0].inference(x_all.cpu(), subgraph_loader)
        
        x_all = torch.matmul(x_all.to(device), self.weight_out)
        log_memory(self.infer_flag, device, "output-transform")
        return x_all.cpu()
    
    def __repr__(self):
        return '{}(layers={}, num_features={}, num_classes={}, hidden_size={}, gpu={})'.format(
            self.__class__.__name__, self.num_layers, self.num_features, self.num_classes,
            self.hidden_size, self.gpu) + '\n' + str(self.convs)

