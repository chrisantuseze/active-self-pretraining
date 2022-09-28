from utils.method_enum import Method


def train_single_epoch(model, train_loader, criterion, optimizer, method, scheduler=None):
    for step, (images, _) in enumerate(train_loader):

        # examples = []
        # if simCLR:
        #     # subject to finetuning
        #     (x_i, x_j) = images
        #     examples.append(x_i)
        #     examples.append(x_j)

        # elif moCo:
        #     examples.append(images[0])
        #     examples.append(images[1])

        # examples[0] = examples[0].cuda(non_blocking=True)
        # examples[1] = examples[1].cuda(non_blocking=True)

        images[0] = images[0].cuda(non_blocking=True)
        images[1] = images[1].cuda(non_blocking=True)


        loss = None
        if method == Method.SIMCLR:
            # positive pair, with encoding
            h_i, h_j, z_i, z_j = model(images[0], images[1])
            loss = criterion(z_i, z_j)

        elif method == Method.MOCO:
            # compute output
            output, target = model(im_q=images[0], im_k=images[1])
            loss = criterion(output, target)

        else:
            NotImplementedError

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # I honestly don't know what this does
        if scheduler is not None:
            scheduler.step()

        # compute accuracy, display progress and other stuff

def train(args, model, train_loader, criterion, optimizer, method=Method.SIMCLR, scheduler=None):
    for epoch in range(args.start_epoch, args.end_epoch):
        train_single_epoch(model, train_loader, criterion, optimizer, method, scheduler)