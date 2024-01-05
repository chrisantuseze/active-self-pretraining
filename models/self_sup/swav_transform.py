'''
Adapted from the SwAV repo
'''

import torch
from models.self_sup.multicropdataset import MultiCropDataset

class TransformsSwAV():
    def __init__(self, args, batch_size, dir):
        
        # build data
        self.train_dataset = MultiCropDataset(
            args,
            dir,
            args.size_crops,
            args.nmb_crops,
            args.min_scale_crops,
            args.max_scale_crops,
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True
        )

        print(f"The size of the dataset is {len(self.train_dataset)} and the number of batches is {self.train_loader.__len__()} for a batch size of {batch_size}")