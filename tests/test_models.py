import argparse

from optimize_pygs.utils import build_args_from_dict
from optimize_pygs.models import build_model
from optimize_pygs.models.pyg_gcn import GCN
from optimize_pygs.configs import DEFAULT_MODEL_CONFIGS


def get_default_args(model):
    # get default args
    default_dict = DEFAULT_MODEL_CONFIGS[model]
    # add model
    default_dict['model'] = model 
    return build_args_from_dict(default_dict)


def test_pyg_gcn():
    # way1 simple way
    args = get_default_args('pyg_gcn')
    model = GCN.build_model_from_args(args)
    print(model)
    
    # way2: build from ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default="pyg_gcn",
                        help='Model')
    GCN.add_args(parser) # get default args
    args = parser.parse_args()
    args.num_features = 100
    args.num_classes = 12
    model = build_model(args)
    print(model)


def test_pyg_models():
    for model_name in ["pyg15_gcn", "pyg15_ggnn", "pyg15_gat", "pyg15_gaan"]:
        args = get_default_args(model_name)
        model = build_model(args)
        print(model)


if __name__ == "__main__":
    test_pyg_models()
    