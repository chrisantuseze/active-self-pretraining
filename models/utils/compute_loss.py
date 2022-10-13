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

    return loss.item()