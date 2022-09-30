import torch
import torch.nn as nn
from models.backbones.resnet import resnet_backbone
from models.heads.nt_xent import NT_Xent
from models.methods.moco.moco import MoCo
from models.methods.simclr.simclr import SimCLR
from models.methods.simclr.modules.optimizer import load_optimizer
from utils.common import load_model, save_model
from utils.method_enum import Method
from datautils import dataset_enum, cifar10, imagenet
from torch.utils.tensorboard import SummaryWriter


class Pretrainer:
    def __init__(self, 
        args, 
        writer) -> None:

        self.args = args
        self.writer = writer

    def train_single_epoch(self, model, train_loader, criterion, optimizer) -> int:
        loss_epoch = 0
        model.train()
        for step, (images, _) in enumerate(train_loader):
            if torch.cuda.is_available():
                images[0] = images[0].cuda(non_blocking=True)
                images[1] = images[1].cuda(non_blocking=True)


            loss = None
            if self.args.method == Method.SIMCLR.value:
                # positive pair, with encoding
                h_i, h_j, z_i, z_j = model(images[0], images[1])
                loss = criterion(z_i, z_j)

            elif self.args.method == Method.MOCO.value:
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

            self.writer.add_scalar("Loss/train_epoch", loss.item(), self.args.global_step)
            self.args.global_step += 1

            loss_epoch += loss.item()
        return loss_epoch

            # compute accuracy, display progress and other stuff

    def pretrain(self, encoder) -> None:

        # initialize ResNet
        print("=> creating model '{}'".format(self.args.resnet))

        n_features = encoder.fc.in_features  # get dimensions of fc layer

        if self.args.reload:
            model.load_state_dict(load_model(self.args))
            
        model = model.to(self.args.device)

        if self.args.method == Method.SIMCLR.value:
            criterion = NT_Xent(self.args.batch_size, self.args.temperature, self.args.world_size)
            model =  SimCLR(encoder, self.args.projection_dim, n_features)
            optimizer, scheduler = load_optimizer(self.args, model)
            print("using SIMCLR")
            
        elif self.args.method == Method.MOCO.value:
            # define loss function (criterion) and optimizer
            criterion = nn.CrossEntropyLoss().cuda(self.args.gpu)
            optimizer = torch.optim.SGD(model.parameters(), self.args.lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
                            
            scheduler = None
            model = MoCo(encoder, self.args.moco_dim, self.args.moco_k, self.args.moco_m, self.args.moco_t, self.args.mlp)
            print("using MOCO")

        else:
            NotImplementedError

        if self.args.dataset == dataset_enum.DatasetType.IMAGENET.value:
            train_loader = imagenet.ImageNet(self.args).get_loader()

        elif self.args.dataset == dataset_enum.DatasetType.CIFAR10.value:
            train_loader = cifar10.CIFAR10(self.args).get_loader()

        else:
            NotImplementedError

        print("Training in progress, please wait...")

        resume_epoch = self.args.current_epoch if self.args.resume else int(self.args.epoch_num)
        end_epoch = self.args.epochs - resume_epoch

        for epoch in range(self.args.start_epoch, end_epoch):
            loss_epoch = self.train_single_epoch(model, train_loader, criterion, optimizer)

            # I honestly don't know what this does
            if scheduler is not None:
                scheduler.step()

            if epoch % 10 == 0:
                save_model(self.args, model, optimizer)

            self.writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            self.writer.add_scalar("Misc/learning_rate", self.args.lr, epoch)
            print(
                f"Epoch [{epoch + resume_epoch}/{end_epoch}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(self.args.lr, 5)}"
            )
            self.args.current_epoch += 1

        save_model(self.args, model, optimizer)