import argparse

from optimize_pygs.models import build_model_from_name
from optimize_pygs.models.configs import DEFAULT_MODEL_CONFIGS


def test_model(model_name):
    # get default args
    model = build_model_from_name(model_name, **DEFAULT_MODEL_CONFIGS[model_name])
    print(model_name, model)


def test_all_models():
    for model_name in ["pyg_gcn", "pyg_gat", "pyg_sage", "pyg15_gcn", "pyg15_ggnn", "pyg15_gat", "pyg15_gaan"]:
        test_model(model_name)        


if __name__ == "__main__":
    test_all_models()
    