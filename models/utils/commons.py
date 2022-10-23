import torch.nn as nn
import torch
import gc

from models.heads.nt_xent import NT_Xent
from models.methods.moco.moco import MoCo
from models.methods.simclr.simclr import SimCLR
from utils.method_enum import Method


def compute_loss(args, images, model, criterion):
    images[0] = images[0].to(args.device)
    images[1] = images[1].to(args.device)

    loss = None
    if args.method == Method.SIMCLR.value:
        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(images[0], images[1])
        loss = criterion(z_i, z_j)

    elif args.method == Method.MOCO.value:
        # compute output
        output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)

    elif args.method == Method.SWAV.value:
        NotImplementedError

    else:
        NotImplementedError

    return loss

def get_model_criterion(args, encoder, isAL=False):
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    if args.method == Method.SIMCLR.value:
        criterion = NT_Xent(args.al_batch_size if isAL else args.batch_size, args.temperature, args.world_size)
        model = SimCLR(encoder, args.projection_dim, n_features)
        print("using SIMCLR")
        
    elif args.method == Method.MOCO.value:
        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().to(args.device)
        model = MoCo(encoder, n_features, args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
        print("using MOCO")

    elif args.method == Method.SWAV.value:
        NotImplementedError

    else:
        NotImplementedError

    return model, criterion

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_params_to_update(model, feature_extract):
    params_to_update = model.parameters()
    print("Params to learn:")

    if feature_extract:
        params_to_update = []

        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                # print("\t",name)
    else:
        None
        # no need to do anything, just update all the params

        # for name, param in model.named_parameters():
        #     if param.requires_grad == True:
        #         print("\t",name)

    return params_to_update

def free_mem(X, y):
    del X
    del y
    gc.collect()
    torch.cuda.empty_cache()