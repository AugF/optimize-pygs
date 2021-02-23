import random
from optimize_pygs.criterions import build_criterion_from_name


def test_criterion(criterion):
    early_stopping = build_criterion_from_name(criterion)
    early_stopping.reset()

    for epoch in range(10):
        val_loss, val_acc = random.random(), random.random()
        print(f"epoch: {epoch}, val_loss: {val_loss}, val_acc: {val_acc}")
        if early_stopping.should_stop(epoch, val_loss, val_acc, epoch):
            break    

    print(early_stopping.get_best_model())


def test_all_criterions():
    for criterion in ["no_stopping_with_acc", "no_stopping_with_loss", "gcn", "gat", "kdd", "gat_with_tolerance"]:
        print(criterion)
        test_criterion(criterion)


if __name__ == "__main__":
    test_all_criterions()


