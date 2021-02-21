增加一个模型的步骤:

1. 创建一个模型
```python
from . import BaseModel, register_model


@register_model("template_model")
class ModelName(BaseModel):
    """
    [Model Name] [Link to paper]
    """
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--batch-size", type=int, default=20)
        # 划分这里可以归到由数据集本身提供
        parser.add_argument("--train-ratio", type=float, default=0.7)
        parser.add_argument("--test-ratio", type=float, default=0.1)
        parser.add_argument("--lr", type=float, default=0.001)
    
    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
        )
    
    def __init__(self, in_feats, hidden_dim, out_feats, k=20, dropout=0.5):
        # 这里k是什么意思
        super(ModelName, self).__init__
        
    def forward(self, x, adj):
        return 0, 0
```

2. 在`__init__.py`文件下注册
```python
SUPPORED_MODELS = {
    ...,
    "template_model": "optimize_pygs.models.template_model", # added
    ...,
}
```

3. optimize_pygs目录下configs.py文件中增加默认配置
```
DEFAULT_MODEL_CONFIGS = {
    ...,
    'template_model': { # added
        "num_features": 100,
        "num_classes": 12,
        "hidden_size": 64,
    }
    ...,
}
```

4. 测试

```python
from optimize_pygs.utils import build_args_from_dict
from optimize_pygs.models import build_model
from optimize_pygs.configs import DEFAULT_MODEL_CONFIGS

def get_default_args(model):
    # get default args
    default_dict = DEFAULT_MODEL_CONFIGS[model]
    # add model
    default_dict['model'] = model 
    return build_args_from_dict(default_dict)

model_name = "template_model"
args = get_default_args(model_name)
model = build_model(args)
print(model)
```