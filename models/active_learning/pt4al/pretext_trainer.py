import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
import numpy as np

from datautils.image_loss_data import Image_Loss
from datautils.target_pretrain_dataset import get_target_pretrain_ds
from models.active_learning.pt4al.pretext_dataloader import PretextDataLoader
from models.backbones.resnet import resnet_backbone

from models.utils.commons import compute_loss, free_mem, get_model_criterion
from utils.commons import load_image_loss, load_saved_state, save_image_loss, simple_load_model, simple_save_model

class PretextTrainer():
    def __init__(self, args, encoder) -> None:
        self.args = args
        self.encoder = encoder
        self.criterion = None

    def train_proxy(self, samples, model, optimizer):

        # convert samples to loader
        loader = PretextDataLoader(self.args, samples).get_loader()

        model.train()
        for epoch in range(self.args.al_epochs):
            for step, (images, _) in enumerate(loader):
                loss = compute_loss(self.args, images, model, self.criterion)
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss = loss.item()
                if step % 50 == 0:
                    print(f"Step [{step}/{len(loader)}]\t Loss: {loss}")

                
    def finetune(self, model, samples):
        # Train using 70% of the samples with the highest loss. So this should be the source of the data
        loader = PretextDataLoader(self.args, samples, finetune=True).get_loader()

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
            
        # Save these images for use during the target pretraining
        return samples[idx[:2000]]

    def make_batches(self):
        # This is a hack to the model can use a batch size of 1 to compute the loss for all the samples
        batch_size = self.args.al_batch_size
        self.args.al_batch_size = 1

        device = torch.device('cpu')

        # model, criterion = get_model_criterion(self.args, self.encoder, isAL=True)
        # state = load_saved_state(self.args, pretrain_level="1")
        # model.load_state_dict(state['model'], strict=False)

        #@TODO remember to remove this and uncomment the lines above
        encoder = resnet_backbone(self.args.al_backbone, pretrained=True)
        model, criterion = get_model_criterion(self.args, encoder, isAL=True)
        criterion = nn.CrossEntropyLoss().to(device)

        model = model.to(device)
        loader = get_target_pretrain_ds(self.args, isAL=True).get_loader()

        model.eval()
        image_loss = []

        print("About to begin eval to make batches")
        with torch.no_grad():
            for step, (images, _) in enumerate(loader):
                images[0] = images[0].to(device)
                images[1] = images[1].to(device)

                # positive pair, with encoding
                h_i, h_j, z_i, z_j = model(images[0], images[1])
                loss = criterion(z_i, z_j)
                
                loss = loss.item()
                if step % 50 == 0:
                    print(f"Step [{step}/{len(loader)}]\t Loss: {loss}")

                image_loss.append(Image_Loss(images[0], images[1], loss))

        sorted_samples = sorted(image_loss, reverse=True)
        save_image_loss(self.args, sorted_samples)

        self.args.al_batch_size = batch_size

        return sorted_samples

    def do_active_learning(self):
        image_loss = load_image_loss(self.args)
        
        if image_loss is None:
            image_loss = self.make_batches()

        proxy_model, self.criterion = get_model_criterion(self.args, self.encoder, isAL=True)
        proxy_model = proxy_model.to(self.args.device)

        optimizer = SGD(proxy_model.parameters(), lr=self.args.al_lr, momentum=0.9, weight_decay=self.args.al_weight_decay)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[160])

        pretraining_sample_pool = []

        for batch in range(self.args.al_batches):
            sample6000 = image_loss[batch * 6000 : (batch + 1) * 6000]

            if batch > 0:
                print('>> Getting previous checkpoint')
                proxy_model.load_state_dict(simple_load_model(f'proxy_{batch-1}.pth'))

                # sampling
                sample2k = self.finetune(proxy_model, sample6000)
            else:
                # first iteration: sample 4k at even intervals
                sample2k = sample6000[2000:]

            pretraining_sample_pool.extend(sample2k)

            if batch < self.args.al_batches - 1: # I want this not to happen for the last iteration since it would be needless
                self.train_proxy(pretraining_sample_pool, proxy_model, optimizer)
                simple_save_model(self.args, proxy_model, f'proxy_{batch}.pth')
                scheduler.step()

        return pretraining_sample_pool

            
