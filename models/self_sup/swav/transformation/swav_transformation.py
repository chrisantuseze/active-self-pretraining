
import torch
from models.self_sup.swav.transformation.multicropdataset import MultiCropDataset


class TransformsSwAV():
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """
    
    def __init__(self, args):
        # build data
        self.train_dataset = MultiCropDataset(
            args.data_path,
            args.size_crops,
            args.nmb_crops,
            args.min_scale_crops,
            args.max_scale_crops,
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True
        )