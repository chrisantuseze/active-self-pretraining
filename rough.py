#!/usr/bin/env python3
import sys

import zipfile
with zipfile.ZipFile("tiny-imagenet-200.zip", 'r') as zip_ref:
    zip_ref.extractall("imagenet")


# self.dir = self.args.dataset_dir + "/" + get_dataset_enum(self.args.target_dataset)

# if is_val:
# val_path_loss_list = []
# img_paths = glob.glob(self.dir + '/test/*/*')[0:len(path_loss_list)]
# for path in img_paths:
#     val_path_loss_list.append(PathLoss(path, 0))     

# self.path_loss_list = val_path_loss_list   