import utils.logger as logging
from sched import scheduler
import numpy as np
import torch
import torch.distributed as dist

from models.backbones.resnet import resnet_backbone
from models.self_sup.myow.loss.cosine_loss import CosineLoss
from models.self_sup.myow.model import MYOW
from models.self_sup.myow.model.mlp3 import MLP3
from models.self_sup.myow.trainer.byol_trainer import BYOLTrainer
from models.self_sup.myow.transformation.transformations import TransformsMYOW
from optim.optimizer import load_optimizer
from models.utils.commons import get_params
from models.utils.training_type_enum import TrainingType
from utils.commons import load_saved_state


class MYOWTrainer(BYOLTrainer):
    def __init__(self, args, writer, encoder, trainingType, pretrain_level, rebuild_al_model=False, train_dataloader=None, view_pool_dataloader=None, 
                transform=None, transform_m=None, view_miner_k=4, decay='cosine', n_decay=1.5, m_decay='cosine', exclude_bias_and_bn=False, 
                symmetric_loss=True, log_step=0, untransform_vis=None):

        self.args = args
        self.writer = writer

        self.world_size = self.args.world_size
        self.device = self.args.device
        self.trainingType = trainingType
        self.train_params = get_params(self.args, self.trainingType)

        # view pool dataloader
        self.view_pool_dataloader = view_pool_dataloader

        # view miner
        self.view_miner_k = view_miner_k

        # transform class for minning
        self.transform_m = transform_m

        # these are on gpu transforms! can have cpu transform in dataloaders
        self.transform_1 = transform # class 1 of transformations
        self.transform_2 = transform # class 2 of transformations

        # build byol trainer
        self.projection_size_2 = 64
        self.projection_hidden_size_2 = 1024

        warmup_epochs = 20
        rampup_epochs = 40
        max_weight = 1.0

        # build network
        self.representation_size = 1000
        self.projection_size = 256
        self.projection_hidden_size = 4096

        # myow loss
        self.mined_loss_weight = 0.
        self.myow_max_weight = max_weight
        self.warmup_epochs = warmup_epochs if warmup_epochs is not None else 0
        self.rampup_epochs = rampup_epochs if rampup_epochs is not None else self.args.epochs

        # convert to steps
        world_size = self.args.world_size
        self.num_examples = len(train_dataloader.dataset)
        self.train_batch_size = self.train_params.batch_size
        self.global_batch_size = world_size * self.train_batch_size
        self.warmup_steps = self.warmup_epochs * self.num_examples // self.global_batch_size
        self.rampup_steps = self.rampup_epochs * self.num_examples // self.global_batch_size
        self.total_steps = self.train_params.epochs * self.num_examples // self.global_batch_size
        

        # logger
        self.untransform_vis = untransform_vis

        state = None
        if trainingType == TrainingType.ACTIVE_LEARNING and not rebuild_al_model:
            self.model = encoder

        else:
            self.model = self.build_model(encoder)

            if trainingType != TrainingType.BASE_PRETRAIN or self.args.epoch_num != self.args.base_epochs:
                state = load_saved_state(self.args, pretrain_level=pretrain_level)
                self.model.load_state_dict(state['model'], strict=False)

        self.model = self.model.to(self.args.device)

        # dataloaders
        self.train_dataloader = train_dataloader

        # transformers
        # these are on gpu transforms! can have cpu transform in dataloaders
        self.trainsform = transform

        self.step = 0
        base_lr = self.train_params.lr / 256
        self.max_lr = base_lr * self.global_batch_size
        self.base_mm = self.args.momentum

        assert decay in ['cosine', 'poly']
        self.decay = decay
        self.n_decay = n_decay

        assert m_decay in ['cosine', 'cste']
        self.m_decay = m_decay

        # configure optimizer
        self.momentum = self.args.momentum
        self.weight_decay = self.args.weight_decay
        self.exclude_bias_and_bn = exclude_bias_and_bn

        if self.exclude_bias_and_bn:
            params = self._collect_params(self.model.trainable_modules)
        else:
            params = self.model.parameters()

        self.optimizer, self.scheduler = load_optimizer(self.args, params, state, self.train_params)

        self.loss = CosineLoss().to(self.device)
        self.symmetric_loss = symmetric_loss

        # logging
        self.log_step = log_step

    def build_model(self, encoder):
        projector_1 = MLP3(self.representation_size, self.projection_size, self.projection_hidden_size)
        projector_2 = MLP3(self.projection_size, self.projection_size_2, self.projection_hidden_size_2)
        predictor_1 = MLP3(self.projection_size, self.projection_size, self.projection_hidden_size)
        predictor_2 = MLP3(self.projection_size_2, self.projection_size_2, self.projection_hidden_size_2)
        net = MYOW(encoder, projector_1, projector_2, predictor_1, predictor_2, n_neighbors=self.view_miner_k)
        return net.to(self.device)

    def update_mined_loss_weight(self, step):
        max_w = self.myow_max_weight
        min_w = 0.
        if step < self.warmup_steps:
            self.mined_loss_weight = min_w
        elif step > self.rampup_steps:
            self.mined_loss_weight = max_w
        else:
            self.mined_loss_weight = min_w + (max_w - min_w) * (step - self.warmup_steps) / \
                                     (self.rampup_steps - self.warmup_steps)

    def log_schedule(self, loss):
        super().log_schedule(loss)
        self.writer.add_scalar('myow_weight', self.mined_loss_weight, self.step)

    def log_correspondance(self, view, view_mined):
        """ currently only implements 2d images"""
        img_batch = np.zeros((16, view.shape[1], view.shape[2], view.shape[3]))
        for i in range(8):
            img_batch[i] = self.untransform_vis(view[i]).detach().cpu().numpy()
            img_batch[8+i] = self.untransform_vis(view_mined[i]).detach().cpu().numpy()
        self.writer.add_images('correspondence', img_batch, self.step)

    def prepare_views(self, inputs):
        x, labels = inputs
        outputs = {'view1': x, 'view2': x}
        return outputs

    def train_epoch(self):
        self.model.train()
        if self.view_pool_dataloader is not None:
            view_pooler = iter(self.view_pool_dataloader)
        for inputs in self.train_dataloader:
            # update parameters
            self.update_learning_rate(self.step)
            self.update_momentum(self.step)
            self.update_mined_loss_weight(self.step)

            # Clear gradients w.r.t. parameters
            self.optimizer.zero_grad()

            inputs = self.prepare_views(inputs) # outputs view1 and view2 (pre-gpu-transform)
            view1 = inputs['view1'].to(self.device)
            view2 = inputs['view2'].to(self.device)

            if self.transform_1 is not None:
                # apply transforms
                view1 = self.transform_1(view1)
                view2 = self.transform_2(view2)

            # Forward pass to get output/logits
            outputs = self.model({'online_view': view1, 'target_view':view2})
            weight = 1 / (1. + self.mined_loss_weight)
            if self.symmetric_loss:
                weight /= 2.

            # Calculate Loss: softmax --> cross entropy loss
            loss = weight * self.forward_loss(outputs['online_q'], outputs['target_z'])

            # Getting gradients w.r.t. parameters
            if self.mined_loss_weight > 0 and not self.symmetric_loss:
                with self.model.no_sync():
                    loss.backward()
            else:
                loss.backward()

            if self.symmetric_loss:
                outputs = self.model({'online_view': view2, 'target_view': view1})
                weight = 1 / (1. + self.mined_loss_weight) / 2.
                loss = weight * self.forward_loss(outputs['online_q'], outputs['target_z'])
                if self.mined_loss_weight > 0:
                    with self.model.no_sync():
                        loss.backward()
                else:
                    loss.backward()

            # mine view
            if self.mined_loss_weight > 0:
                if self.view_pool_dataloader is not None:
                    try:
                        # currently only supports img, label
                        view_pool, label_pool = next(view_pooler)
                        view_pool = view_pool.to(self.device).squeeze()
                    except StopIteration:
                        # reinit the dataloader
                        view_pooler = iter(self.view_pool_dataloader)
                        view_pool, label_pool = next(view_pooler)
                        view_pool = view_pool.to(self.device).squeeze()
                    view3 = inputs['view1'].to(self.device)
                else:
                    view3 = inputs['view3'].to(self.device).squeeze() \
                        if 'view3' in inputs else inputs['view1'].to(self.device).squeeze()
                    view_pool = inputs['view_pool'].to(self.device).squeeze()

                # apply transform
                if self.transform_m is not None:
                    # apply transforms
                    view3 = self.transform_m(view3)
                    view_pool = self.transform_m(view_pool)

                # compute representations
                outputs = self.model({'online_view': view3}, get_embedding='encoder')
                online_y = outputs['online_y']
                outputs_pool = self.model({'target_view': view_pool}, get_embedding='encoder')
                target_y_pool = outputs_pool['target_y']

                # mine views
                selection_mask = self.model.mine_views(online_y, target_y_pool)

                target_y_mined = target_y_pool[selection_mask].contiguous()
                outputs_mined = self.model({'online_y': online_y,'target_y': target_y_mined}, get_embedding='predictor_m')
                weight = self.mined_loss_weight / (1. + self.mined_loss_weight)
                loss = weight * self.forward_loss(outputs_mined['online_q_m'], outputs_mined['target_v'])
                loss.backward()

            # Updating parameters
            self.optimizer.step()

            # update moving average
            self.update_target_network()

            # log
            if self.step % self.log_step == 0:
                # self.log_schedule(loss=loss.item())
                logging.info(f"Step [{self.step}/{len(self.train_dataloader)}]\t Loss: {loss}")

            # log images
            if self.mined_loss_weight > 0 and self.log_img_step > 0 and self.step % self.log_img_step == 0 and self.rank == 0:
                self.log_correspondance(view3, view_pool[selection_mask])

            # update parameters
            self.step += 1

        return loss.item()



def get_myow_trainer(args, writer, encoder, dataloader, pretrain_level, rebuild_al_model=False, trainingType=TrainingType.BASE_PRETRAIN):
    params = get_params(args, trainingType)

    transformMYOW = TransformsMYOW(params.image_size)
    transform, transform_m = transformMYOW.transform, transformMYOW.transform_m
    
    trainer = MYOWTrainer(args, writer, encoder, trainingType, pretrain_level, rebuild_al_model,
                        train_dataloader=dataloader, view_pool_dataloader=dataloader, transform=transform,
                        transform_m=transform_m, exclude_bias_and_bn=True, 
                        symmetric_loss=True, view_miner_k=1,
                        decay='cosine', m_decay='cosine', log_step=500)

    return trainer