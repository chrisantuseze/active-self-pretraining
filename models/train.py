import torch
from utils.common import save_model
from utils.method_enum import Method
from torch.utils.tensorboard import SummaryWriter


def train_single_epoch(args, model, train_loader, criterion, optimizer, writer):
    loss_epoch = 0
    for step, (images, _) in enumerate(train_loader):
        if torch.cuda.is_available():
            images[0] = images[0].cuda(non_blocking=True)
            images[1] = images[1].cuda(non_blocking=True)


        loss = None
        if args.method == Method.SIMCLR.value:
            # positive pair, with encoding
            h_i, h_j, z_i, z_j = model(images[0], images[1])
            loss = criterion(z_i, z_j)

        elif args.method == Method.MOCO.value:
            # compute output
            output, target = model(im_q=images[0], im_k=images[1])
            loss = criterion(output, target)

        else:
            NotImplementedError

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
        args.global_step += 1

        loss_epoch += loss.item()
    return loss_epoch

        # compute accuracy, display progress and other stuff

def train(args, model, train_loader, criterion, optimizer, scheduler=None):
    print("Training in progress, please wait...")

    writer = SummaryWriter()
    for epoch in range(args.start_epoch, args.epochs):
        loss_epoch = train_single_epoch(args, model, train_loader, criterion, optimizer, writer)

        # I honestly don't know what this does
        if scheduler is not None:
            scheduler.step()

        if epoch % 10 == 0:
            save_model(args, model, optimizer)

        writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
        writer.add_scalar("Misc/learning_rate", args.lr, epoch)
        print(
            f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(args.lr, 5)}"
        )
        args.current_epoch += 1