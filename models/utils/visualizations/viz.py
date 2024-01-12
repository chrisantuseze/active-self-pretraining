import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch

from models.backbones.resnet import resnet_backbone
from datautils.dataset_enum import get_dataset_info
from datautils.target_dataset import get_pretrain_ds
from models.utils.training_type_enum import TrainingType
from utils.commons import load_chkpts, load_saved_state, simple_load_model
import models.self_sup.swav.backbone.resnet50 as resnet_models


def visualize_features_(args, model, source_data, target_data):
    source_features_extended = []
    for step, (images, targets) in enumerate(source_data):
        with torch.no_grad():
            images = images.to(args.device)
            source_features = model(images)
            source_features_extended.append(source_features)

        if len(source_features_extended) > 128:
            break
    
    source_features = torch.cat(source_features_extended)

    target_features_extended = []
    for step, (images, targets) in enumerate(target_data):
        with torch.no_grad():
            images = images.to(args.device)
            target_features = model(images)
            target_features_extended.append(target_features)

        if len(target_features_extended) > 128:
            break

    target_features = torch.cat(target_features_extended)

    # Assuming 'features' is your high-dimensional feature matrix
    print(source_features.shape, target_features.shape)
    num_samples, num_features = source_features.shape

    # Choose a perplexity value less than the number of samples
    perplexity_value = min(64, num_samples - 1)  # You can adjust this value

    # Create t-SNE object with specified perplexity
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)

    # Extract features before adaptation
    source_emb = tsne.fit_transform(source_features) 
    target_emb = tsne.fit_transform(target_features)

    plt.scatter(source_emb[:,0], source_emb[:,1], c='b', label='Source')
    plt.scatter(target_emb[:,0], target_emb[:,1], c='r', label='Target')
    plt.legend()
    # plt.title('Before Adaptation')
    plt.show()

def visualize_source_model_features(args, source_model, source_data, target_data):
    for _, (images, _) in enumerate(source_data):
        with torch.no_grad():
            images = images.to(args.device)
            source_features = source_model(images)
        break

    for _, (images, _) in enumerate(target_data):
        with torch.no_grad():
            images = images.to(args.device)
            target_features = source_model(images)
        break

    # Assuming 'features' is your high-dimensional feature matrix
    num_samples, num_features = source_features.shape

    # Choose a perplexity value less than the number of samples
    perplexity_value = min(64, num_samples - 1)  # You can adjust this value

    # Create t-SNE object with specified perplexity
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)

    # Extract features before adaptation
    source_emb = tsne.fit_transform(source_features) 
    target_emb = tsne.fit_transform(target_features)

    # Create a new figure for each iteration
    plt.figure()

    plt.scatter(source_emb[:,0], source_emb[:,1], c='b', label='Source')
    plt.scatter(target_emb[:,0], target_emb[:,1], c='r', label='Target')
    plt.legend()
    plt.title('Before Adaptation')
    plt.savefig('models/utils/visualizations/plots/source_model_plot.png')

    # plt.show()

def visualize_adapted_model_features(args, adapted_model, source_data, target_data, batch):
    for _, (images, _) in enumerate(source_data):
        with torch.no_grad():
            images = images.to(args.device)
            adapted_source_features = adapted_model(images)
        break

    for _, (images, _) in enumerate(target_data):
        with torch.no_grad():
            images = images.to(args.device)
            adapted_target_features = adapted_model(images)
        break

    # Assuming 'features' is your high-dimensional feature matrix
    num_samples, num_features = adapted_source_features.shape

    # Choose a perplexity value less than the number of samples
    perplexity_value = min(64, num_samples - 1)  # You can adjust this value

    # Create t-SNE object with specified perplexity
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)

    # Extract features after adaptation
    source_emb = tsne.fit_transform(adapted_source_features)
    target_emb = tsne.fit_transform(adapted_target_features) 

    # Create a new figure for each iteration
    plt.figure()

    plt.scatter(source_emb[:,0], source_emb[:,1], c='b', label='Source')
    plt.scatter(target_emb[:,0], target_emb[:,1], c='r', label='Target')
    plt.legend()
    plt.title(f'After Adaptation -> batch {batch}')

    # Save figure to image file
    plt.savefig(f'models/utils/visualizations/plots/adapted_model_plot_{batch}.png')

    # plt.show()

def visualize_features_both(args, source_model, adapted_model, source_data, target_data):

    for _, (images, _) in enumerate(source_data):
        with torch.no_grad():
            images = images.to(args.device)
            source_features = source_model(images)
        break

    for _, (images, _) in enumerate(target_data):
        with torch.no_grad():
            images = images.to(args.device)
            target_features = source_model(images)
        break

    for _, (images, _) in enumerate(source_data):
        with torch.no_grad():
            images = images.to(args.device)
            adapted_source_features = adapted_model(images)
        break

    for _, (images, _) in enumerate(target_data):
        with torch.no_grad():
            images = images.to(args.device)
            adapted_target_features = adapted_model(images)
        break

    # Assuming 'features' is your high-dimensional feature matrix
    print(source_features.shape, target_features.shape)
    num_samples, num_features = source_features.shape

    # Choose a perplexity value less than the number of samples
    perplexity_value = min(64, num_samples - 1)  # You can adjust this value

    # Create t-SNE object with specified perplexity
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)

    # Extract features before adaptation
    source_emb = tsne.fit_transform(source_features) 
    target_emb = tsne.fit_transform(target_features)

    plt.scatter(source_emb[:,0], source_emb[:,1], c='b', label='Source')
    plt.scatter(target_emb[:,0], target_emb[:,1], c='r', label='Target')
    plt.legend()
    plt.title('Before Adaptation')
    plt.show()

    # Extract features after adaptation
    source_emb = tsne.fit_transform(adapted_source_features)
    target_emb = tsne.fit_transform(adapted_target_features) 

    plt.scatter(source_emb[:,0], source_emb[:,1], c='b', label='Source')
    plt.scatter(target_emb[:,0], target_emb[:,1], c='r', label='Target')
    plt.legend()
    plt.title('After Adaptation')
    plt.show()

def viz(args):
    os.makedirs('models/utils/visualizations/plots/', exist_ok=True)
    args.swav_batch_size = 256
    encoder = resnet_backbone(args.backbone, pretrained=False)

    # encoder = resnet_models.__dict__[args.backbone](
    #         zero_init_residual=True,
    #         normalize=True,
    #         hidden_mlp=args.hidden_mlp,
    #         output_dim=args.feat_dim,
    #         nmb_prototypes=args.nmb_prototypes,
    #     )

    args.base_dataset = 5
    args.target_dataset = 6

    source_loader = get_pretrain_ds(args, training_type=TrainingType.BASE_PRETRAIN).get_loader() 
    target_loader = get_pretrain_ds(args, training_type=TrainingType.TARGET_PRETRAIN).get_loader() 

    # num_classes, source_ds_name, dir = get_dataset_info(args.base_dataset)
    # source_model = encoder
    # state = load_saved_state(args, dataset=source_ds_name, pretrain_level="1")
    # source_model.load_state_dict(state['model'], strict=False)
    # source_model = source_model.to(args.device)
    # source_model.eval()

    # visualize_source_model_features(args, source_model, source_loader, target_loader)

    _, target_ds_name, _ = get_dataset_info(args.target_dataset)
    for batch in range(args.al_batches):
        target_model = encoder
        # state = simple_load_model(args, "1_finetuner_dslr.pth")

        state = load_saved_state(args, dataset=target_ds_name, pretrain_level=f"2_{batch}")
        target_model.load_state_dict(state['model'], strict=False)
        target_model = target_model.to(args.device)
        target_model.eval()

        visualize_adapted_model_features(args, target_model, source_loader, target_loader, batch)

    # Ensure that figures are closed at the end of the loop
    plt.close('all')