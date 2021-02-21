## Introduction

## Usage

```
class CustomDataset:
    ...
    def get_evaluator(self):
        evaluator = get_evaluator(self.metric)
        if evaluator == None:
            return NotImplementedError
        return evaluator

    def get_loss_fn(self):
        loss_fn = get_loss_fn(self.metric)
        if loss_fn == None:
            return NotImplementedError
        return loss_fn
```

## TODO

[ ] 引入更多真实的数据集