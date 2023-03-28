from sys import prefix
import torch
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler

import os
import argparse
import time
from datautils.dataset_enum import get_dataset_enum

import models.gan1.visualizers as visualizers
from models.utils.commons import get_params, AverageMeter, get_params_to_update
from models.gan1.dataloaders.setup_dataloader_smallgan import setup_dataloader 
from models.gan1.nets.setup_model import setup_model
from models.gan1.losses.AdaBIGGANLoss import AdaBIGGANLoss
from utils.commons import load_chkpts, simple_load_model, simple_save_model
import utils.logger as logging
import models.self_sup.swav.backbone.resnet50 as resnet_models


def argparse_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="face", help = "dataset. anime or face. ")
    parser.add_argument('--pretrained', type=str, default="./save/checkpoints/G_ema.pth", help = "pretrained BigGAN model")


    parser.add_argument('--eval-freq', type=int, default=500, help = "save frequency in iteration. currently no eval is implemented and just model saving and sample generation is performed" )
    parser.add_argument('--gpu', '-g', type=str, default='-1')
    
    #learning rates
    parser.add_argument('--lr-g-l', type=float, default=0.0000001, help = "lr for original linear layer in generator. default is 0.0000001")
    parser.add_argument('--lr-g-batch-stat', type=float, default=0.0005, help = "lr for linear layer to generate scale and bias in generator")
    parser.add_argument('--lr-embed', type=float, default=0.05, help = "lr for image embeddings")
    parser.add_argument('--lr-bsa-l', type=float, default=0.0005, help = "lr for statistic (scale and bias) parameter for the original fc layer in generator. This is newly intoroduced learnable parameter for the pretrained GAN")
    parser.add_argument('--lr-c-embed', type=float, default=0.001, help = "lr for class conditional embeddings")
    
    
    #loss settings
    parser.add_argument('--loss-per', type=float, default=0.1, help = "scaling factor for perceptural loss. ")
    parser.add_argument('--loss-emd', type=float, default=0.1, help = "scaling factor for earth mover distance loss. ")
    parser.add_argument('--loss-re', type=float, default=0.02, help = "scaling factor for regularizer loss. ")
    parser.add_argument('--loss-norm-img', type=int, default=1, help = "normalize img loss or not")
    parser.add_argument('--loss-norm-per', type=int, default=1, help = "normalize perceptural loss or not")
    parser.add_argument('--loss-dist-per', type=str, default="l2", help = "distance function for perceptual loss. l1 or l2")

    parser.add_argument('--step', type=int, default=3000, help="Decrease lr by a factor of args.step_facter every <step> iterations")
    parser.add_argument('--step-facter', type=float, default=0.1, help="facter to multipy when decrease lr ")

    parser.add_argument('--iters', type=int, default=10000, help="number of iterations.")
    parser.add_argument('--batch', type=int, default=25, help="batch size")
    parser.add_argument('--workers', type=int, default=4, help="number of processes to make batch worker. default is 8")
    parser.add_argument('--model', type=str,default = "biggan128-ada", help = "model. biggan128-ada")

    parser.add_argument('--resume', type=str, default=None, help="model weights to resume")
    parser.add_argument('--savedir',  default = "train", help='Output directory')
    parser.add_argument('--saveroot',  default = "./experiments", help='Root directory to make the output directory')

    parser.add_argument('-p', '--print-freq', default=500, type=int, help='print frequency ')

    parser.add_argument('--checkpoint_path', type=str, default="./datasets/generated", help='model checkpoint path')
    
    return parser.parse_args()


def do_gen_ai(args):
    device = args.device

    dataset = get_dataset_enum(args.target_dataset)

    gen_images_path = os.path.join(args.dataset_dir, f'{args.base_dataset}')
    # gen_images_path = os.path.join(args.dataset_dir, f'generated_{dataset}')

    if not os.path.exists(gen_images_path):
        os.makedirs(gen_images_path)
    
    logging.info(f"Using {dataset} dataset")

    dir = f'{args.dataset_dir}/{dataset}'
    dataloader = setup_dataloader(dir=dir, batch_size=args.gan_batch_size, num_workers=args.workers)
    dataset_size = len(dataloader.dataset)
    
    model_path = os.path.join(args.model_checkpoint_path, args.pretrained)
    model = setup_model(args.model_name, dataset_size=dataset_size, resume=args.gan_resume, model_path=model_path)
    model.eval()
    #this has to be eval() even if it's training time
    #because we want to fix batchnorm running mean and var
    #still tune batchnrom scale and bias that is generated by linear layer in biggan
    
    optimizer, scheduler = setup_optimizer(model,
                                lr_g_linear=args.lr_g_l,
                                lr_g_batch_stat=args.lr_g_batch_stat,
                                lr_bsa_linear=args.lr_bsa_l,
                                lr_embed=args.lr_embed,
                                lr_class_cond_embed=args.lr_c_embed,
                                step=args.step,
                                step_factor=args.step_factor,
                            )
    ''
    criterion = AdaBIGGANLoss(
                    scale_per=args.loss_per,
                    scale_emd=args.loss_emd,
                    scale_reg=args.loss_re,
                    normalize_img=args.loss_norm_img,
                    normalize_per=args.loss_norm_per,
                    dist_per=args.loss_dist_per,
                )
    
    #start trainig loop
    losses = AverageMeter()
    batch_time = AverageMeter()
    
    iteration = 0
    epoch = 0

    model = model.to(device)
    criterion = criterion.to(device)
    end = time.time()

    while(True):
        # Iterate over dataset (one epoch).

        for data in dataloader: 
            img, indices = data[0].to(device), data[1].to(device)
            
            #embeddings (i.e. z) + noise (i.e. epsilon) 
            embeddings = model.embeddings(indices)
            embeddings_eps = torch.randn(embeddings.size(), device=device)*0.01
            #see https://github.com/nogu-atsu/SmallGAN/blob/f604cd17516963d8eec292f3faddd70c227b609a/gen_models/ada_generator.py#L29
            
            #forward
            img_generated = model(embeddings + embeddings_eps)
            loss = criterion(img_generated, img, embeddings, model.linear.weight)
            losses.update(loss.item(), img.size(0))
            batch_time.update(time.time() - end)

            #compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            end = time.time()
            if iteration % args.log_step == 0:
                logging.info(
                    "Epoch: [{0}][{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Lr: {lr:.4f}".format(
                        epoch,
                        iteration,
                        batch_time=batch_time,
                        loss=losses,
                        lr=optimizer.param_groups[0]["lr"],
                    )
                )
                
            if iteration > args.iters:
                break
            iteration +=1
            
        scheduler.step()

        if iteration > args.iters:
            break
        epoch+=1

    simple_save_model(args, model, f'gan_{dataset}_model_{epoch}.pth')

    logging.info("Generating images...")
    img_prefix = os.path.join(gen_images_path, "%d_"%iteration) 
    generate_samples(model, img_prefix, size=400)


def generate_samples(model, img_prefix, size):
    # visualizers.reconstruct(model, img_prefix, num=size, add_small_noise=True)
    # visualizers.interpolate(model, img_prefix, source=0, dist=1, trncate=0.3, num=size)
    for i in range(1, 5):
        visualizers.random(model, img_prefix, tmp=0.3, num=size, prefix=i, truncate=True)

def setup_optimizer(model, lr_g_batch_stat, lr_g_linear, lr_bsa_linear, lr_embed, lr_class_cond_embed, step,   step_factor=0.1):
    #group parameters by lr
    params = []
    params.append({"params": list(model.batch_stat_gen_params().values()), "lr": lr_g_batch_stat})
    params.append({"params": list(model.linear_gen_params().values()), "lr": lr_g_linear })
    params.append({"params": list(model.bsa_linear_params().values()), "lr": lr_bsa_linear })
    params.append({"params": list(model.emebeddings_params().values()), "lr": lr_embed })
    params.append({"params": list(model.calss_conditional_embeddings_params().values()), "lr": lr_class_cond_embed})
    
    #setup optimizer
    optimizer = optim.Adam(params, lr=0)#0 is okay because sepcific lr is set by `params`
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step, gamma=step_factor)
    return optimizer, scheduler

def standalone_image_gen(args):
    model_path = os.path.join(args.model_checkpoint_path, args.pretrained)
    model = setup_model(args.model_name, dataset_size=25, resume=args.gan_resume, model_path=model_path)

    # state = simple_load_model(args, 'gan_model_10000.000000.pth')
    # model.load_state_dict(state['model'], strict=False)
    model = model.to(args.device)

    gen_images_path = os.path.join(args.dataset_dir, f'{args.base_dataset}')
    if not os.path.exists(gen_images_path):
        os.makedirs(gen_images_path)

    logging.info(f"The path to save the generated images is {gen_images_path}")

    generate_samples(model, f'{gen_images_path}/', size=400)
    
if __name__ == '__main__':
    gan_args = argparse_setup()
    do_gen_ai(gan_args)