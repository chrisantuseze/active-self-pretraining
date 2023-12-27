import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from datautils.dataset_enum import get_dataset_enum
from datautils.target_dataset import get_pretrain_ds

from models.trainers.resnet import resnet_backbone
from models.utils.commons import get_ds_num_classes
from models.utils.training_type_enum import TrainingType
from utils.commons import simple_load_model

def visualize_features(args, source_model, adapted_model, source_data, target_data):
    for step, (images, targets) in enumerate(source_data):
        images = images.to(args.device)
        source_features = source_model(images)

    for step, (images, targets) in enumerate(target_data):
        images = images.to(args.device)
        source_target_features = source_model(images)

    for step, (images, targets) in enumerate(target_data):
        images = images.to(args.device)
        target_features = adapted_model(images)

    # Extract features before adaptation
    # source_features = source_model.features(source_data)
    # source_target_features = source_model.features(target_data)

    tsne = TSNE(n_components=2)
    source_emb = tsne.fit_transform(source_features) 
    target_emb = tsne.fit_transform(source_target_features)

    plt.scatter(source_emb[:,0], source_emb[:,1], c='b', label='Source')
    plt.scatter(target_emb[:,0], target_emb[:,1], c='r', label='Target')
    plt.legend()
    plt.title('Before Adaptation')

    # Extract features after adaptation
    # source_features = adapted_model.features(source_data) 
    # target_features = adapted_model.features(target_data)

    source_emb = tsne.fit_transform(source_features)
    target_emb = tsne.fit_transform(target_features) 

    plt.scatter(source_emb[:,0], source_emb[:,1], c='b', label='Source')
    plt.scatter(target_emb[:,0], target_emb[:,1], c='r', label='Target')
    plt.legend()
    plt.title('After Adaptation')

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

    target_model = encoder
    state = simple_load_model(args, path=f'target_{get_dataset_enum(args.target_dataset)}.pth')
    target_model.load_state_dict(state['model'], strict=False)
    target_model = target_model.to(args.device)
    target_model.eval()

    source_train_loader, _ = get_pretrain_ds(args, training_type=TrainingType.SOURCE_PRETRAIN).get_loaders() 
    target_train_loader, _ = get_pretrain_ds(args, training_type=TrainingType.TARGET_PRETRAIN).get_loaders() 

    visualize_features(source_model, target_model, source_train_loader, target_train_loader)

    # for step, (images, targets) in enumerate(source_train_loader):
    #     images = images.to(args.device)
    #     source_features = source_model(images)

    # for step, (images, targets) in enumerate(target_train_loader):
    #     images = images.to(args.device)
    #     target_features = target_model(images)