import os
from sys import prefix
import torch

import pickle
from PIL import Image
from models.active_learning.al_method_enum import get_al_method_enum

from models.utils.ssl_method_enum import get_ssl_method
from datautils.dataset_enum import get_dataset_enum
import utils.logger as logging


def save_state(args, model, dataset, pretrain_level="1"):
    if not os.path.isdir(args.model_checkpoint_path):
        os.makedirs(args.model_checkpoint_path)

    out = os.path.join(args.model_checkpoint_path, "swav_{}_checkpoint_{}.tar".format(pretrain_level, dataset))

    state = {'model': model.state_dict()}
    torch.save(state, out)
    print("checkpoint saved at {}".format(out))

def load_saved_state(args, pretrain_level="1"):
    try:
        dataset = get_dataset_enum(args.target_dataset)
        out = os.path.join(args.model_checkpoint_path, "swav_{}_checkpoint_{}.tar".format(pretrain_level, dataset))

        logging.info(f"Loading checkpoint from - {out}")
        return torch.load(out, map_location=args.device.type)

    except IOError as er:
        logging.error(er)
        return None

def load_classifier_chkpts(args, model, pretrain_level="1"):
    dataset = get_dataset_enum(args.target_dataset)
    filename = "sawv_{}_checkpoint_{}.tar".format(pretrain_level, dataset)
    return load_chkpts(args, filename, model)

def load_chkpts(args, filename, model):
    try:
        out = os.path.join(
            args.model_checkpoint_path, filename
        )
    
        state_dict = torch.load(out, map_location="cuda:0")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # remove prefixe "module."
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        for k, v in model.state_dict().items():
            if k not in list(state_dict):
                pass
            elif state_dict[k].shape != v.shape:
                state_dict[k] = v
        msg = model.load_state_dict(state_dict, strict=False)

        return model

    except IOError as er:
        logging.error(er)
        return None

def simple_save_model(args, model, path):
    state = {
        'model': model.state_dict()
    }

    out = os.path.join(args.model_checkpoint_path, path)
    torch.save(state, out)

def simple_load_model(args, path):
    try:
        out = os.path.join(args.model_checkpoint_path, path)
        return torch.load(out)

    except IOError as er:
        return None

def accuracy(pred, target, topk=1):
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


def save_path_loss(args, filename, image_loss_list):
    filename = "{}_{}".format(get_dataset_enum(args.target_dataset), filename)
    out = os.path.join(args.model_misc_path, filename)

    try:
        with open(out, "wb") as file:
            pickle.dump(image_loss_list, file)

        logging.info(f"path loss saved at {out}")

    except IOError as er:
        None


def load_path_loss(args, filename):
    filename = "{}_{}".format(get_dataset_enum(args.target_dataset), filename)
    out = os.path.join(args.model_misc_path, filename)

    try:
        with open(out, "rb") as file:
            return pickle.load(file)

    except IOError as er:
        return None

def save_accuracy_to_file(args, accuracies, best_accuracy, filename):
    out = os.path.join(args.model_misc_path, filename)

    try:
        with open(out, "a") as file:
            file.write("The accuracies are: \n")
            file.write(", ".join(accuracies))

            file.write("\nThe best accuracy is: \n")
            file.write(str(best_accuracy))

            logging.info(f"accuracies saved saved at {out}")

    except IOError as er:
        None

def load_accuracy_file(args):
    dataset = f"{get_dataset_enum(args.base_dataset)}-{get_dataset_enum(args.target_dataset)}-{get_dataset_enum(args.finetune_dataset)}"
    filename = "{}_{}_batch_{}.txt".format(dataset, get_al_method_enum(args.al_method), args.finetune_epochs)
    out = os.path.join(args.model_misc_path, filename)

    try:
        with open(out, "a") as file:
            return file.readlines()

    except IOError as er:
        return None

def save_class_names(args, label):
    filename = f"{get_dataset_enum(args.target_dataset)}.txt"
    out = os.path.join(args.model_misc_path, filename)

    try:
        with open(out, "a") as file:
            file.write(f"{str(label)}\n")

    except IOError as er:
        logging.error(er)
        None

def load_class_names(args):
    filename = f"{get_dataset_enum(args.target_dataset)}.txt"
    out = os.path.join(args.model_misc_path, filename)

    try:
        with open(out) as file:
            return file.readlines()

    except IOError as er:
        logging.error(er)
        return None

def pil_loader(path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

def get_accuracy_file_ext(args):
    if args.do_gradual_base_pretrain and args.base_pretrain:
        return f'_{args.al_trainer_sample_size}'

    return ''


def get_state_for_da(args,  pretrain_level=2):
    dataset = get_dataset_enum(args.base_dataset)

    filename = "swav_{}_checkpoint_{}.tar".format(pretrain_level, dataset)
    logging.info(f"Loading [uc2] checkpoint from - {filename}")

    return simple_load_model(args, path=filename)