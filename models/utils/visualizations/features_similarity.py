import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datautils.target_dataset import get_target_pretrain_ds

from models.backbones.resnet import resnet_backbone
from models.utils.commons import AverageMeter
from models.utils.training_type_enum import TrainingType
from utils.commons import load_chkpts
import utils.logger as logging


class FeatureSimilarity():
    def __init__(self, args) -> None:
        self.args = args
        encoder = resnet_backbone(self.args.backbone, pretrained=False)
        self.model = load_chkpts(self.args, "swav_800ep_pretrain.pth.tar", encoder)

    def visualize_features(self):
        loader1, loader2 = self.get_loaders()
        
        model1 = self.train_model(self.model, loader1)
        model2 = self.train_model(self.model, loader2)

        latents1 = self.get_reps(model1, loader1)
        latents2 = self.get_reps(model2, loader2)

        self.visualize_latent_reps(latents1)
        self.visualize_latent_reps(latents2)
        

    def train_model(self, model, loader):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        model.train()
        end = time.time()
        total_steps = 0
        for epoch in range(50):
            for step, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3) in enumerate(loader):
            
                inputs, inputs1 = inputs.to(self.args.device), inputs1.to(self.args.device)
                targets, targets1 = targets.to(self.args.device), targets1.to(self.args.device)
                inputs2, inputs3 = inputs2.to(self.args.device), inputs3.to(self.args.device)
                targets2, targets3 = targets2.to(self.args.device), targets3.to(self.args.device)

                optimizer.zero_grad()
                outputs, outputs1, outputs2, outputs3 = model(inputs), model(inputs1), model(inputs2), model(inputs3)

                loss = criterion(outputs, targets)
                loss1 = criterion(outputs1, targets1)
                loss2 = criterion(outputs2, targets2)
                loss3 = criterion(outputs3, targets3)
                loss_avg = (loss + loss1 + loss2 + loss3) / 4.
                loss_avg.backward()
                optimizer.step()

                losses.update(loss_avg.item(), inputs[0].size(0))
                batch_time.update(time.time() - end)
                end = time.time()

                if step % self.args.log_step == 0:
                    logging.info(
                        "Epoch: [{0}][{1}]\t"
                        "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "Lr: {lr:.4f}".format(
                            epoch, step,
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses, lr=optimizer.param_groups[0]["lr"],
                        )
                    )

                total_steps = step
        avg_loss = losses.sum/total_steps
            
        return model

    def get_reps(self, model, loader):
        model.eval()
        latent_reps = []

        with torch.no_grad():
            for step, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3) in enumerate(loader):
                inputs, inputs1, targets, targets1 = inputs.to(self.args.device), inputs1.to(self.args.device), targets.to(self.args.device), targets1.to(self.args.device)
                inputs2, inputs3, targets2, targets3 = inputs2.to(self.args.device), inputs3.to(self.args.device), targets2.to(self.args.device), targets3.to(self.args.device)
                
                outputs = model(inputs)
                outputs1 = model(inputs1)
                outputs2 = model(inputs2)
                outputs3 = model(inputs3)

                outputs = torch.cat([outputs, outputs1, outputs2, outputs3], dim=0)
                latent_reps.append(outputs)
        return torch.cat(latent_reps, dim=0)

    def visualize_latent_reps(self, latent_reps, labels=None):
        tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=0)
        latent_reps_2d = tsne.fit_transform(latent_reps)
        plt.scatter(latent_reps_2d[:, 0], latent_reps_2d[:, 1])#, c=labels)
        plt.colorbar()

    def calculate_similarity(latent_reps_1, latent_reps_2):
        return F.cosine_similarity(latent_reps_1, latent_reps_2, dim=1)

    def get_loaders(self):
        loader1, test_loader = get_target_pretrain_ds(
            self.args, training_type=TrainingType.BASE_PRETRAIN).get_finetuner_loaders(
                train_batch_size=self.args.al_finetune_batch_size,
                val_batch_size=100
            )

        loader2, test_loader = get_target_pretrain_ds(
            self.args, training_type=TrainingType.ACTIVE_LEARNING).get_finetuner_loaders(
                train_batch_size=self.args.al_finetune_batch_size,
                val_batch_size=100
            )

        return loader1, loader2
        