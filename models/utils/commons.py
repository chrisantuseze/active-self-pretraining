import torch.nn as nn
import torch
import gc

from models.heads.nt_xent import NT_Xent
from models.self_sup.simclr.simclr import SimCLR
from models.utils.ssl_method_enum import SSL_Method
from models.utils.training_type_enum import Params, TrainingType


def compute_loss(args, images, model, criterion):
    images[0] = images[0].to(args.device)
    images[1] = images[1].to(args.device)

    loss = None
    if args.method == SSL_Method.SIMCLR.value:
        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(images[0], images[1])
        loss = criterion(z_i, z_j)

    else:
        NotImplementedError

    return loss

def compute_loss_for_al(args, images, model, criterion):
    images[0] = images[0].to(args.device)
    images[1] = images[1].to(args.device)

    loss, output1, output2 = None, None, None
    if args.method == SSL_Method.SIMCLR.value:
        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(images[0], images[1])
        output1, output2 = z_i, z_j

        loss = criterion(z_i, z_j)

    else:
        NotImplementedError

    return loss, output1, output2

def get_model_criterion(args, encoder, training_type=TrainingType.ACTIVE_LEARNING):
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    if args.method == SSL_Method.SIMCLR.value:

        params = get_params(args, training_type)
        batch_size = params.batch_size

        criterion = NT_Xent(batch_size, args.temperature, args.world_size)
        model = SimCLR(encoder, args.projection_dim, n_features)
        print("using SIMCLR")

    else:
        NotImplementedError

    return model, criterion

def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

def get_params_to_update(model, feature_extract):
    params_to_update = model.parameters()

    if feature_extract:
        params_to_update = []

        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                # print("\t", name)
    else:
        None
        # no need to do anything, just update all the params

        # for name, param in model.named_parameters():
        #     if param.requires_grad == True:
        #         print("\t",name)

    return params_to_update

def get_params(args, training_type):
    params = {
        TrainingType.ACTIVE_LEARNING: Params(batch_size=args.al_batch_size, image_size=args.al_image_size, lr=args.al_lr, epochs=args.al_epochs),
        TrainingType.AL_FINETUNING: Params(batch_size=args.al_finetune_batch_size, image_size=args.al_image_size, lr=args.al_lr, epochs=args.al_epochs),
        TrainingType.BASE_PRETRAIN: Params(batch_size=args.base_batch_size, image_size=args.base_image_size, lr=args.base_lr, epochs=args.base_epochs),
        TrainingType.TARGET_PRETRAIN: Params(batch_size=args.target_batch_size, image_size=args.target_image_size, lr=args.target_lr, epochs=args.target_epochs),
        TrainingType.FINETUNING: Params(batch_size=args.finetune_batch_size, image_size=args.finetune_image_size, lr=args.finetune_lr, epochs=args.finetune_epochs),
    }
    return params[training_type]

def accuracy(loss, corrects, loader):
        epoch_loss = loss / len(loader.dataset)
        epoch_acc = corrects.double() / len(loader.dataset)

        return epoch_loss, epoch_acc

def free_mem(X, y):
    del X
    del y
    gc.collect()
    torch.cuda.empty_cache()