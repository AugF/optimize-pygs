import importlib

from optimize_pygs.models.base_model import BaseModel

MODEL_REGISTRY = {}

def register_model(name):
    """
    New model types can be added with the :func:`register_model`
    function decorator.
    
    For example:
        @register_model('gat')
        class GAT(BaseModel):
            (...)
    """
    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Cannot register duplicate model ({name})")
        if not issubclass(cls, BaseModel):
            raise ValueError(f"Model ({name}: {cls.__name__}) must extend BaseModel")
        MODEL_REGISTRY[name] = cls
        cls.model_name = name
        return cls 
    return register_model_cls


def try_import_model(model):
    if model not in MODEL_REGISTRY:
        if model in SUPPORED_MODELS:
            importlib.import_module(SUPPORED_MODELS[model])
        else:
            print(f"Failed to import {model} model.")
            return False
    return True

def build_model(args):
    if not try_import_model(args.model):
        exit(1)
    return MODEL_REGISTRY[args.model].build_model_from_args(args)
        
        
SUPPORED_MODELS = {
    "template_model": "optimize_pygs.models.template_model",
    "pyg_gcn": "optimize_pygs.models.pyg_gcn",
    "pyg_gat": "optimize_pygs.models.pyg_gat",
    "pyg_sage": "optimize_pygs.models.pyg_sage",
    "pyg15_gcn": "optimize_pygs.models.pyg15_gcn",
    "pyg15_ggnn": "optimize_pygs.models.pyg15_ggnn",
    "pyg15_gat": "optimize_pygs.models.pyg15_gat",
    "pyg15_gaan": "optimize_pygs.models.pyg15_gaan",
}