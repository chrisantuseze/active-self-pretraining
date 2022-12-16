'''
Adapted from the SwAV repo
'''

import torch
from models.self_sup.swav.transformation.multicropdataset import MultiCropDataset

class TransformsSwAV():
    def __init__(self, args, batch_size, dir=None, pathloss_list=None):
        # build data
        self.train_dataset = MultiCropDataset(
            dir,
            args.size_crops,
            args.nmb_crops,
            args.min_scale_crops,
            args.max_scale_crops,
            pathloss_list,
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True
        )