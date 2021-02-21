import os.path as osp

from optimize_pygs.global_configs import dataset_root
from optimize_pygs.utils import download_url, accuracy, multilabel_f1, bce_with_logits_loss, cross_entropy_loss
from optimize_pygs.datasets import CustomDataset, register_dataset


@register_dataset("ppi")
class PPIDataset(CustomDataset):
    def __init__(self):
        dataset = "ppi"
        super(PPIDataset, self).__init__(dataset_root, dataset)

    def get_evaluator(self):
        return multilabel_f1

    def get_loss_fn(self):
        return bce_with_logits_loss


@register_dataset("ppi-large")
class PPILargeDataset(CustomDataset):
    def __init__(self):
        dataset = "ppi-large"
        super(PPILargeDataset, self).__init__(dataset_root, dataset)

    def get_evaluator(self):
        return multilabel_f1

    def get_loss_fn(self):
        return bce_with_logits_loss


@register_dataset("flickr")
class FlickrDatset(CustomDataset):
    def __init__(self):
        dataset = "flickr"
        super(FlickrDatset, self).__init__(dataset_root, dataset)

    def get_evaluator(self):
        return accuracy

    def get_loss_fn(self):
        return cross_entropy_loss
    
    
@register_dataset("reddit")
class RedditDataset(CustomDataset):
    def __init__(self):
        dataset = "reddit"
        super(RedditDataset, self).__init__(dataset_root, dataset)
        
    def get_evaluator(self):
        return accuracy

    def get_loss_fn(self):
        return cross_entropy_loss
    
    
@register_dataset("yelp")        
class YelpDataset(CustomDataset):
    def __init__(self, transform=None, pretransform=None):
        dataset = "yelp"
        super(YelpDataset, self).__init__(dataset_root, dataset)

    def get_evaluator(self):
        return multilabel_f1

    def get_loss_fn(self):
        return bce_with_logits_loss


@register_dataset("amazon")
class AmazonDataset(CustomDataset):
    def __init__(self):
        dataset = "amazon"
        super(AmazonDataset, self).__init__(dataset_root, dataset)

    def get_evaluator(self):
        return multilabel_f1

    def get_loss_fn(self):
        return bce_with_logits_loss
    
    
def test_yelp():
    data = YelpDataset()
    print(data[0])

if __name__ == "__main__":
    test_yelp()