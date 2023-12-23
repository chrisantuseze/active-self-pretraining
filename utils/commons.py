import os
import torch

import pickle
from PIL import Image
from models.active_learning.al_method_enum import get_al_method_enum

from datautils.dataset_enum import get_dataset_enum
import utils.logger as logging


def save_state(args, model, optimizer, pretrain_level="1", optimizer_type="Adam-Cosine"):
    if not os.path.isdir(args.model_checkpoint_path):
        os.makedirs(args.model_checkpoint_path)

    prefix = None#get_ssl_method(args.method)
    dataset = get_dataset_enum(args.target_dataset)

    additional_ext = get_accuracy_file_ext(args)
    logging.info(f"Appending the extension, {additional_ext}")

    out = os.path.join(args.model_checkpoint_path, "{}_{}_checkpoint_{}_{}{}.tar".format(prefix, pretrain_level, dataset, args.current_epoch, additional_ext))

    state = {
        'model': model.state_dict(),
        optimizer_type + '-optimizer': optimizer.state_dict()
    }
    torch.save(state, out)

    print("checkpoint saved at {}".format(out))

def load_saved_state(args, recent=True, pretrain_level="1"):
    try:
        prefix = None #get_ssl_method(args.method)
        if pretrain_level == "2":
            epoch_num = args.target_epochs

        else:
            epoch_num = args.source_epochs

        dataset = get_dataset_enum(args.target_dataset)
        additional_ext = get_accuracy_file_ext(args)
        logging.info(f"Appending the extension, {additional_ext}")

        out = os.path.join(
                args.model_checkpoint_path, "{}_{}_checkpoint_{}_{}{}.tar".format(prefix, pretrain_level, dataset, epoch_num, additional_ext)
            )

        logging.info(f"Loading checkpoint from - {out}")
        return torch.load(out, map_location=args.device.type)

    except IOError as er:
        logging.error(er)
        return None

def load_classifier_chkpts(args, model, pretrain_level="1"):
    prefix = None #get_ssl_method(args.method)
    if pretrain_level == "2":
        epoch_num = args.target_epochs

    else:
        epoch_num = args.source_epochs

    dataset = get_dataset_enum(args.target_dataset)
    filename = "{}_{}_checkpoint_{}_{}.tar".format(prefix, pretrain_level, dataset, epoch_num)
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
    dataset = f"{get_dataset_enum(args.source_dataset)}-{get_dataset_enum(args.target_dataset)}"
    filename = "{}_{}_batch.txt".format(dataset, get_al_method_enum(args.al_method))
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
    # if args.do_gradual_base_pretrain and args.base_pretrain:
    #     return f'_{args.al_trainer_sample_size}'

    return ''


def get_state_for_da(args):
    prefix = None #"swav"

    pretrain_level = 2
    epoch_num = args.target_epochs
    dataset = get_dataset_enum(args.source_dataset)

    filename = "{}_{}_checkpoint_{}_{}.tar".format(prefix, pretrain_level, dataset, epoch_num)
    logging.info(f"Loading [uc2] checkpoint from - {filename}")

    return simple_load_model(args, path=filename)