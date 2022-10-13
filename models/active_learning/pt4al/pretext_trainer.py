import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
import numpy as np

from datautils.finetune_dataset import Finetune
from datautils.image_loss_data import Image_Loss
from datautils.target_pretrain_dataset import get_target_pretrain_ds

from models.backbones.resnet import resnet_backbone
from models.utils.compute_loss import compute_loss
from utils.common import load_saved_state, simple_load_model, simple_save_model
from utils.method_enum import Method

class PretextTrainer():
    def __init__(self, args) -> None:
        self.args = args
        self.criterion = nn.CrossEntropyLoss().to(args.device)
        self.loss_gen_loader = None
        self.finetune_loader = None

    def train_proxy(self, samples, model, optimizer):

        # convert samples to loader
        loader = None
        model.train()
        for epoch in range(100):
            print("Proxy epoch - {}".format(epoch))

            for _, (images, _) in enumerate(loader):
                loss = compute_loss(self.args, images, model, self.criterion)
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                # Show progress here

    def finetune(self, model, samples):
        # Train using 70% of the samples with the highest loss. So this should be the source of the data
        loader = Finetune(self.args, samples)

        # sub5k = Loader2(is_train=False,  transform=transform_test, path_list=samples)
        # ploader = torch.utils.data.DataLoader(sub5k, batch_size=1, shuffle=False, num_workers=2)

        top1_scores = []
        model.eval()
        with torch.no_grad():
            for _, (images, _) in enumerate(loader):
                images[0] = images[0].to(self.args.device)
                images[1] = images[1].to(self.args.device)

                outputs = model(images[0])
                scores, predicted = outputs.max(1)

                # save top1 confidence score 
                outputs = F.normalize(outputs, dim=1)
                probs = F.softmax(outputs, dim=1)
                top1_scores.append(probs[0][predicted.item()])

        idx = np.argsort(top1_scores)
        samples = np.array(samples)
            
        # Save this images for use during the target pretraining

    def make_batches(self):
        encoder = resnet_backbone(self.args.resnet, pretrained=False)
        state = load_saved_state(self.args, pretrain_level="1")
        encoder.load_state_dict(state['model'])

        loader = get_target_pretrain_ds(self.args).get_loader()

        encoder.eval()

        image_loss = []

        with torch.no_grad():
            for _, (images, _) in enumerate(loader):
                loss = compute_loss(self.args, images, encoder, self.criterion)

                #TODO save this loss so it can be batched and used in the main task
                image_loss.append(Image_Loss(images[0], images[1], loss))

        
        return None # So either return the data in batches or return it as a single batch

    def do_active_learning(self):
        proxy_model = resnet_backbone(self.args.al_backbone, pretrained=False)
        proxy_model = proxy_model.to(self.args.device)

        optimizer = SGD(proxy_model.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[160])

        image_loss = self.make_batches()

        for batch in range(self.args.al_batches):
            sample5000 = None # image_loss[] TODO Get 5000 per batch. This number is determined by the data budget that was made

            if batch > 0:
                print('>> Getting previous checkpoint')
                proxy_model.load_state_dict(simple_load_model(f'proxy_{batch-1}.pth'))

                # sampling
                sample4k = self.finetune(proxy_model, sample5000)
            else:
                # first iteration: sample 1k at even intervals
                samples = np.array(samples)
                sample4k = samples[[j*5 for j in range(1000)]]

            self.train_proxy(sample4k, proxy_model, optimizer)
            simple_save_model(self.args, proxy_model, f'proxy_{batch}.pth')
            scheduler.step()

            
