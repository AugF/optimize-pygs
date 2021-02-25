from neuroc_pygs.configs import EXP_DATASET
from neuroc_pygs.options import get_args, build_dataset


for data in EXP_DATASET:
    args = get_args()
    args.dataset = data
    try:
        data = build_dataset(args)
        print(args.dataset, data)
    except Exception as e:
        print(e)