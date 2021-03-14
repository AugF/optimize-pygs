# 待做
from optimize_pygs.data.graph_saint import GraphSAINTRandomWalkSampler, GraphSAINTEdgeSampler, GraphSAINTNodeSampler
from neuroc_pygs.options import get_args, build_dataset
# 剪枝: 对于图采样, 运用采样算法; 对于层采样，运用neighbor的sample方法

# 这里有问题，如何考虑剪枝
args = get_args()
data = build_dataset(args)

loader1 = GraphSAINTNodeSampler(data, batch_size=6000, shuffle=True,
                                     num_steps=5, 
                                     num_workers=4) # batch size

# loader2 = GraphSAINTEdgeSampler(data, batch_size=6000, 
#                                      num_steps=5, sample_coverage=100,
#                                      num_workers=4) # batch_size

# loader3 = GraphSAINTRandomWalkSampler(data, batch_size=6000, walk_length=2,
#                                      num_steps=5, sample_coverage=100,
#                                      num_workers=4) # batch_size walk length

# get Data对象
# print(iter(loader1).next()) 
# print(iter(loader2).next())
# print(iter(loader2).next())