import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from datautils.finetune_dataset import Finetune
from models.backbones.resnet import resnet_backbone
from models.heads.logloss_head import LogLossHead
from utils.commons import accuracy, load_saved_state


class Classifier:
    def __init__(self,
                args,
                pretrained=None) -> None:
        self.args = args
        self.encoder = resnet_backbone(self.args.resnet, pretrained=False)
        self.model = LogLossHead(self.encoder, with_avg_pool=True, in_channels=2048, num_classes=None) #todo: The num_classes parameter is determined by the dataset used for the finetuning
        
        self.model = self.model.to(self.args.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)

        self.criterion = nn.CrossEntropyLoss().to(self.args.device)
        self.saved_state = load_saved_state(args, pretrain_level="2")
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None) -> None:
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print('load model from: {}'.format(pretrained))
        self.encoder.init_weights(pretrained=pretrained)
        # self.head.init_weights() @todo: Thoroughly check what this is and what it does

        self.model.load_state_dict(self.state['model'])

        # freeze layers and remove output

    def finetune(self) -> None:
        self.model = self.model.to(self.args.device)

        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)

        train_loader, val_loader = Finetune(self.args)

        for epoch in range(self.args.finetune_start_epoch, self.args.finetune_epochs):

            # train for one epoch
            self.train_single_epoch(train_loader, self.model, self.criterion, self.optimizer, epoch)

            # evaluate on validation set
            self.validate(val_loader, self.model, self.criterion)
            
            scheduler.step()

            
            # remember best acc@1 and save checkpoint
            # is_best = acc1 > best_acc1
            # best_acc1 = max(acc1, best_acc1)


    def train_single_epoch(self, train_loader, model, criterion, optimizer, epoch) -> None:
        model.train()

        # end = time.time()
        for i, (images, target) in enumerate(train_loader):

            images = images.to(self.args.device)
            target = target.to(self.args.device)
                
            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc = accuracy(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def validate(self, val_loader, model, criterion) -> None:    
        model.eval()

        with torch.no_grad():
            for i, (images, target) in enumerate(val_loader):

                images = images.to(self.args.device)
                target = target.to(self.args.device)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc = accuracy(output, target, topk=(1, 5))

        return #top1.avg

