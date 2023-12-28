import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch

from datautils.dataset_enum import get_dataset_enum
from datautils.target_dataset import get_pretrain_ds

from models.trainers.resnet import resnet_backbone
from models.utils.commons import get_ds_num_classes
from models.utils.training_type_enum import TrainingType
from utils.commons import simple_load_model

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

    plt.scatter(source_emb[:,0], source_emb[:,1], c='b', label='Source')
    plt.scatter(target_emb[:,0], target_emb[:,1], c='r', label='Target')
    plt.legend()
    plt.title('Before Adaptation')
    plt.show()

def visualize_adapted_model_features(args, adapted_model, source_data, target_data):
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

    plt.scatter(source_emb[:,0], source_emb[:,1], c='b', label='Source')
    plt.scatter(target_emb[:,0], target_emb[:,1], c='r', label='Target')
    plt.legend()
    plt.title('After Adaptation')
    plt.show()

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

def visualize_classification_accuracy(source_model, adapted_model, source_data, target_data):
    domain_classifier = DomainClassifier() 

    # Before adaptation
    source_features = source_model.features(source_data)
    source_target_features = source_model.features(target_data)

    src_acc = domain_classifier.accuracy(source_features) 
    tgt_acc = domain_classifier.accuracy(source_target_features)

    print('Before Adaptation:') 
    print(f'Source accuracy: {src_acc:.2f}')
    print(f'Target accuracy: {tgt_acc:.2f}')

    # After adaptation
    source_features = adapted_model.features(source_data)
    target_features = adapted_model.features(target_data) 

    src_acc = domain_classifier.accuracy(source_features)
    tgt_acc = domain_classifier.accuracy(target_features) 

    print('After Adaptation:')
    print(f'Source accuracy: {src_acc:.2f}') 
    print(f'Target accuracy: {tgt_acc:.2f}')


def viz(args):
    num_classes, dir = get_ds_num_classes(args.source_dataset)
    encoder = resnet_backbone(args.backbone, num_classes, pretrained=False)
    
    source_model = encoder
    state = simple_load_model(args, path=f'source_{get_dataset_enum(args.source_dataset)}.pth')
    source_model.load_state_dict(state['model'], strict=False)
    source_model = source_model.to(args.device)
    source_model.eval()

    source_train_loader, _ = get_pretrain_ds(args, training_type=TrainingType.SOURCE_PRETRAIN).get_loaders() 
    target_train_loader, _ = get_pretrain_ds(args, training_type=TrainingType.TARGET_PRETRAIN).get_loaders() 

    visualize_source_model_features(args, source_model, source_train_loader, target_train_loader)

    for batch in range(args.al_batches):
        target_model = encoder
        state = simple_load_model(args, path=f'target_{get_dataset_enum(args.target_dataset)}{str(batch-1)}.pth')
        target_model.load_state_dict(state['model'], strict=False)
        target_model = target_model.to(args.device)
        target_model.eval()

        visualize_adapted_model_features(args, target_model, source_train_loader, target_train_loader)