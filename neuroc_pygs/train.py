import copy
import numpy as np
from neuroc_pygs.train_step import train, test, infer
from neuroc_pygs.options import get_args, build_dataset, build_model


args = get_args()
print(args)
data, train_loader, subgraph_loader = build_dataset(args)
model, optimizer = build_model(args, data) 
model, data = model.to(args.device), data.to(args.device)


# step1 fit
best_val_acc = 0
best_model = None
for epoch in range(args.epochs):
    train_acc = train(model, data, train_loader, optimizer, args.mode, args.device)
    if args.infer_layer:
        val_acc, _ = infer(model, data, subgraph_loader, args.device, split="val")
    else:
        val_acc, _ = test(model, data, subgraph_loader, args.device, split="val") # 这里好像没什么用
    print(f"Epoch: {epoch:03d}, Accuracy: Train: {train_acc:.4f}, Val: {val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = copy.deepcopy(model)

# step2 predict
if args.infer_layer:
    test_acc, _ = infer(best_model, data, subgraph_loader, args.device, split="val")
else:
    test_acc, _ = test(best_model, data, subgraph_loader, args.device, split="val") 
print(f"final test acc: {test_acc:.4f}")
