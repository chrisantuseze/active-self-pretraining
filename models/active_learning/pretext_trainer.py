import torch
import torch.nn.functional as F
import time
import numpy as np
from datautils.dataset_enum import get_dataset_info
from models.trainers.selfsup_pretrainer import SelfSupPretrainer
from optim.optimizer import load_optimizer
import utils.logger as logging
from typing import List
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import copy

from datautils.path_loss import PathLoss
from datautils.target_dataset import get_pretrain_ds
from models.active_learning.pretext_dataloader import PretextDataLoader
from models.backbones.resnet import resnet_backbone

from models.utils.commons import AverageMeter, get_feature_dimensions_backbone, get_model_criterion, get_params
from models.utils.training_type_enum import TrainingType
from utils.commons import load_path_loss, load_saved_state, save_path_loss, simple_load_model, simple_save_model

class PretextTrainer():
    def __init__(self, args, writer) -> None:
        self.args = args
        self.writer = writer
        self.criterion = None
        self.best_proxy_acc = 0
        self.best_batch = 0
        self.best_trainer_acc = 0
        self.val_acc_history = []
        self.best_model = None
        self.n_pretext_classes = 4

        self.n_features = get_feature_dimensions_backbone(self.args)
        self.num_classes, self.dataset, self.dir = get_dataset_info(self.args.target_dataset)

    def batch_sampler_entropy_only(self, model, samples: List[PathLoss]) -> List[PathLoss]:
        loader = PretextDataLoader(self.args, samples, is_val=True, batch_size=1).get_loader()

        logging.info(f"Generating the top1 scores using")
        _preds = []

        model, _ = get_model_criterion(self.args, model, num_classes=self.n_pretext_classes)

        model = model.to(self.args.device)
        model.eval()
        with torch.no_grad():
            for step, (inputs, _) in enumerate(loader):
                inputs = inputs.to(self.args.device)
                outputs = model(inputs)

                _preds.append(self.get_predictions(outputs))

                if step % self.args.log_step == 0:
                    logging.info(f"Eval Step [{step}/{len(loader)}]")

        preds = torch.cat(_preds).numpy()
       
        return self.get_new_samples_entropy_only(preds, samples)

    def batch_sampler(self, model, samples: List[PathLoss]) -> List[PathLoss]:
        loader = PretextDataLoader(self.args, samples, is_val=True, batch_size=1).get_loader()

        logging.info(f"Generating the sample weights")

        model, _ = get_model_criterion(self.args, model, num_classes=self.n_pretext_classes)
        state = simple_load_model(self.args, path=f'bayesian_model_{self.dataset}.pth')
        model.load_state_dict(state['model'], strict=False)
        model = model.to(self.args.device)
        model.eval()

        # get data embeddings
        embeds = []
        _preds = []
        with torch.no_grad():
            for step, (inputs, _) in enumerate(loader):
                inputs = inputs.to(self.args.device)
                outputs = model(inputs)
                embeds.append(outputs.detach().cpu())
                _preds.append(self.get_predictions(outputs))

        preds = torch.cat(_preds).numpy()
        embeds = np.concatenate(embeds)
       
        # return self.get_new_samples_entropy_only(preds, samples)
        return self.get_new_samples(preds, samples, embeds)
    
    def get_predictions(self, outputs):
        dist1 = F.softmax(outputs, dim=1)
        preds = dist1.detach().cpu()

        return preds

    def get_new_samples_entropy_only(self, preds, samples) -> List[PathLoss]:
        entropy = -(preds * np.log(preds)).sum(axis=1)
        indices = entropy.argsort(axis=0)[::-1] # sort in descending order

        new_samples = []
        for item in indices:
            new_samples.append(samples[item]) # Map back to original indices

        return new_samples[-self.args.sampling_size:]

    def get_new_samples(self, preds, samples, embeds) -> List[PathLoss]:
        preds += 3
        entropy = -(preds * np.log(preds)).sum(axis=1)        
        indices = self.get_diverse(entropy, embeds)

        new_samples = [samples[i] for i in indices]
        return new_samples
    
    def get_diverse(self, entropy, embeds):
        k = self.args.sampling_size
        kmeans = KMeans(n_clusters=k)  
        kmeans.fit(embeds, sample_weight=entropy)

        # Find nearest neighbors to inferred centroids
        dists = euclidean_distances(kmeans.cluster_centers_, embeds)
        sort_idxs = dists.argsort(axis=1)[::-1]

        q_idxs = []
        ax, rem = 0, k
        while rem > 0:
            q_idxs.extend(list(sort_idxs[:, ax][:rem]))
            q_idxs = list(set(q_idxs))
            rem = k-len(q_idxs)
            ax += 1

        return q_idxs
    
    def get_diverse_old(self, entropy, target_embeds, core_set_embeds):
        # Cluster coreset into k clusters 
        kmeans = KMeans(n_clusters=self.n_pretext_classes+2)  
        kmeans.fit(target_embeds, sample_weight=entropy)

        # Get cluster assignment for each coreset point
        coreset_clusters = kmeans.predict(core_set_embeds)

        # Get cluster assignment for each target point
        x_clusters = kmeans.predict(target_embeds)

        # Distances to cluster centroids
        cluster_dists = [] 

        # Distances from target x to coreset
        for i, x in enumerate(target_embeds):
            min_dist = float('inf')
        
            # Get assigned cluster for x
            x_cluster = x_clusters[i]
            
            # Compute distance to closest point in assigned coreset cluster
            for c in core_set_embeds[coreset_clusters == x_cluster]:  
                min_dist = min(min_dist, np.linalg.norm(x - c))


            cluster_dists.append(min_dist) 

        cluster_dists = np.array(cluster_dists)

        return cluster_dists

    def make_batches(self, model, prefix, training_type=TrainingType.ACTIVE_LEARNING):
        loader = get_pretrain_ds(self.args, training_type=training_type, is_train=False, batch_size=1).get_loader()

        model, criterion = get_model_criterion(self.args, model, num_classes=self.n_pretext_classes)
        state = simple_load_model(self.args, path=f'bayesian_model_{self.dataset}.pth')
        model.load_state_dict(state['model'], strict=False)
        model = model.to(self.args.device)

        model.eval()

        test_loss = 0
        correct = 0
        total = 0
        pathloss = []

        logging.info("About to begin eval to make batches")
        with torch.no_grad():
            for step, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3, path) in enumerate(loader):
                inputs, inputs1, targets, targets1 = inputs.to(self.args.device), inputs1.to(self.args.device), targets.to(self.args.device), targets1.to(self.args.device)
                inputs2, inputs3, targets2, targets3 = inputs2.to(self.args.device), inputs3.to(self.args.device), targets2.to(self.args.device), targets3.to(self.args.device)
                outputs = model(inputs)
                outputs1 = model(inputs1)
                outputs2 = model(inputs2)
                outputs3 = model(inputs3)
                loss1 = criterion(outputs, targets)
                loss2 = criterion(outputs1, targets1)
                loss3 = criterion(outputs2, targets2)
                loss4 = criterion(outputs3, targets3)
                loss = (loss1+loss2+loss3+loss4)/4.
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                loss = loss.item()

                if step % self.args.log_step == 0:
                    logging.info(f"Eval Step [{step}/{len(loader)}]\t Loss: {loss}")

                if isinstance(path, tuple) or isinstance(path, list):
                    path = path[0]

                pathloss.append(PathLoss(path, loss))
        
        sorted_samples = sorted(pathloss, key=lambda x: x.loss, reverse=False)#reverse=True)
        save_path_loss(self.args, self.args.al_path_loss_file, sorted_samples)

        return sorted_samples

    def eval_bm(self, model, criterion, test_loader):
        batch_time = AverageMeter()
        losses = AverageMeter()

        model.eval()
        end = time.time()
        total, correct = 0, 0

        total_steps = 0
        with torch.no_grad():
            for step, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3) in enumerate(test_loader):
                inputs, inputs1, targets, targets1 = inputs.to(self.args.device), inputs1.to(self.args.device), targets.to(self.args.device), targets1.to(self.args.device)
                inputs2, inputs3, targets2, targets3 = inputs2.to(self.args.device), inputs3.to(self.args.device), targets2.to(self.args.device), targets3.to(self.args.device)
                
                outputs = model(inputs)
                outputs1 = model(inputs1)
                outputs2 = model(inputs2)
                outputs3 = model(inputs3)

                loss = criterion(outputs, targets)
                loss1 = criterion(outputs1, targets1)
                loss2 = criterion(outputs2, targets2)
                loss3 = criterion(outputs3, targets3)
                loss_avg = (loss + loss1 + loss2 + loss3) / 4.

                _, predicted = outputs.max(1)
                _, predicted1 = outputs1.max(1)
                _, predicted2 = outputs2.max(1)
                _, predicted3 = outputs3.max(1)
                total += targets.size(0)*4

                correct += predicted.eq(targets).sum().item()
                correct += predicted1.eq(targets1).sum().item()
                correct += predicted2.eq(targets2).sum().item()
                correct += predicted3.eq(targets3).sum().item()

                losses.update(loss_avg.item(), inputs[0].size(0))
                
                # measure elapsed time
                batch_time.update(time.perf_counter() - end)
                end = time.perf_counter()

                total_steps = step if step > 0 else 1

        # Save checkpoint.
        epoch_acc = 100. * correct / total

        if epoch_acc > self.best_trainer_acc:
            self.best_model = copy.deepcopy(model)
            self.best_trainer_acc = epoch_acc

        avg_loss = losses.sum/total_steps
        logging.info(
            "Test:\t"
            "Time {batch_time.avg:.3f}\t"
            "Loss {loss:.4f}\t"
            "Acc@1 {top1:.3f}\t"
            "Best Acc@1 so far {acc:.1f}".format(
                batch_time=batch_time, loss=avg_loss, top1=epoch_acc, acc=self.best_trainer_acc))

        return epoch_acc, avg_loss


    def train_bm(self, model, epoch, criterion, optimizer, scheduler, train_loader):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        model.train()
        end = time.time()
        total_steps = 0
        for step, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3) in enumerate(train_loader):
            
            # update learning rate
            scheduler.step(epoch, step)

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

            total_steps = step if step > 0 else 1
        avg_loss = losses.sum/total_steps
            
        logging.info("Train Loss: {:.4f}".format(avg_loss))
        return avg_loss

    def bayesian_model(self, model, prefix, path_list: List[PathLoss]=None, training_type=TrainingType.ACTIVE_LEARNING):
        if path_list is not None:
            path_list = [path.path for path in path_list]
            train_loader, test_loader = get_pretrain_ds(self.args, training_type=training_type).get_bm_loaders(path_list)
        else:
            train_loader, test_loader = get_pretrain_ds(self.args, training_type=training_type).get_bm_loaders()

        model, criterion = get_model_criterion(self.args, model, num_classes=self.n_pretext_classes)
        state = None
        if prefix == "first":
            state = load_saved_state(self.args, dataset=get_dataset_info(self.args.source_dataset)[1], pretrain_level="1")
            model.load_state_dict(state['model'], strict=False)
        model = model.to(self.args.device)

        train_params = get_params(self.args, TrainingType.ACTIVE_LEARNING)
        optimizer, scheduler = load_optimizer(self.args, model.parameters(), train_params=train_params, train_loader=train_loader)

        counter = 0
        epochs = train_params.epochs
        logging.info("Running bayesian model")
        for epoch in range(epochs):
            logging.info('\nEpoch {}/{}'.format(epoch, epochs))
            logging.info('-' * 20)

            _ = self.train_bm(model, epoch, criterion, optimizer, scheduler, train_loader)
            epoch_acc, _ = self.eval_bm(model, criterion, test_loader)

            # update learning rate
            scheduler.step()

            if epoch_acc <= self.best_trainer_acc:
                counter += 1
            else:
                counter = 0

            if counter > 20:
                logging.info("Early stopped at epoch {}:".format(epoch))
                break

        simple_save_model(self.args, self.best_model, f'bayesian_model_{self.dataset}.pth')

    def do_active_learning(self) -> List[PathLoss]:
        logging.info_x(f"Base = {get_dataset_info(self.args.source_dataset)[1]}, Target = {get_dataset_info(self.args.target_dataset)[1]}")
        encoder = resnet_backbone(self.args.backbone, pretrained=False)
        
        state = simple_load_model(self.args, path=f'bayesian_model_{self.dataset}.pth')
        if not state:
            self.bayesian_model(encoder, prefix='first')

        path_loss = load_path_loss(self.args, self.args.al_path_loss_file)
        if path_loss is None:
            path_loss = self.make_batches(encoder, prefix='first')

        return self.active_learning(path_loss, encoder)

    def active_learning(self, path_loss, encoder):
        core_set = []

        path_loss = path_loss[::-1] # this does a reverse active learning to pick only the most certain data

        self.args.sampling_size = int(0.75 * (len(path_loss))//self.args.al_batches)
        logging.info(f"Total size of target dataset is ", len(path_loss))

        sample_per_batch = len(path_loss)//self.args.al_batches
        batch_sampler_encoder = encoder

        for batch in range(self.args.al_batches):
            logging.info_x(f'>> Batch {batch}')

            sampled_data = path_loss[batch * sample_per_batch : (batch + 1) * sample_per_batch]
            # sampling
            samplek = self.batch_sampler(batch_sampler_encoder, sampled_data)
            batch_sampler_encoder = encoder

            core_set.extend(samplek)
            logging.info(f"Size of core-set is {len(core_set)}")

            loader = PretextDataLoader(self.args, core_set, training_type=TrainingType.TARGET_AL).get_loader()
            pretrainer = SelfSupPretrainer(self.args, self.writer)
            pretrainer.source_pretrain(loader, self.args.target_epochs, batch, trainingType=TrainingType.TARGET_AL)

            if batch < self.args.al_batches - 1: # I want this not to happen for the last iteration since it would be needless
                self.bayesian_model(encoder, prefix=str(batch), path_list=core_set, training_type=TrainingType.ACTIVE_LEARNING)

        
        return core_set