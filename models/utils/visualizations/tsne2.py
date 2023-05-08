import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datautils.dataset_enum import get_dataset_enum
from datautils.path_loss import PathLoss

from datautils.target_dataset import get_target_pretrain_ds
from models.active_learning.pretext_dataloader import PretextDataset
from models.utils.commons import get_images_pathlist
from models.utils.training_type_enum import TrainingType

import utils.logger as logging

# # Define a function to extract features from an image using the ResNet-18 model
# def extract_features(image):
#     with torch.no_grad():
#         features = model(image.unsqueeze(0)).squeeze()
#     return features

# Define a function to extract features from the data using a pre-trained model
def extract_features(model, dataset):
    # Create a dataloader to load the data in batches
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1000, 
        shuffle=True, num_workers=4,
        pin_memory=True,)

    logging.info("Extracting features...")

    features = []
    labels = []
    with torch.no_grad():
        for images, target in dataloader:
            # Move the data to the GPU if available
            if torch.cuda.is_available():
                images = images.cuda()
            
            # Extract the features using the pre-trained model
            output = model(images)
            features.append(output.cpu().numpy())
            labels.append(target.numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

def tsne_similarity(args):

    # Load the ResNet-18 model
    model = models.resnet18(pretrained=True)
    model.eval()

    # Remove the last fully connected layer
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model = model.to(args.device)

    # Define a data transformation pipeline
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load images from the three datasets and extract their features
    args.target_dataset = 16
    ds_1 = get_target_pretrain_ds(args, training_type=TrainingType.BASE_PRETRAIN)
    dataset1 = ds_1.get_dataset(transform, is_tsne=True)

    ds = f"generated_{get_dataset_enum(args.target_dataset)}"
    img_path = get_images_pathlist(f'{args.dataset_dir}/{ds}', with_train=True)
    path_loss_list = [PathLoss(path, 0) for path in img_path]
    dataset2 = PretextDataset(args, path_loss_list, transform, False)

    args.target_dataset = 19
    ds_3 = get_target_pretrain_ds(args, training_type=TrainingType.ACTIVE_LEARNING)
    dataset3 = ds_3.get_dataset(transform, is_tsne=True)
    
    dataset1_features = extract_features(model, dataset1)
    dataset2_features = extract_features(model, dataset2)
    dataset3_features = extract_features(model, dataset3)

    # Concatenate the features into a single feature matrix
    features = torch.cat([torch.stack(dataset1_features), torch.stack(dataset2_features), torch.stack(dataset3_features)])

    # Compute the pairwise cosine similarities between the features
    similarities = torch.matmul(features, features.t())
    norms = similarities.norm(dim=1, keepdim=True)
    similarities = similarities / norms / norms.t()

    # Apply t-SNE to the similarities to obtain a 2D embedding
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)

    logging.info("Generating TSNE embeddings...")
    embedding = tsne.fit_transform(similarities.cpu().numpy())

    # Split the embedding back into the three datasets
    dataset1_embedding = embedding[:len(dataset1_features)]
    dataset2_embedding = embedding[len(dataset1_features):len(dataset1_features)+len(dataset2_features)]
    dataset3_embedding = embedding[len(dataset1_features)+len(dataset2_features):]

    # Visualize the t-SNE embedding using a scatter plot
    plt.scatter(dataset1_embedding[:,0], dataset1_embedding[:,1], color='red', label='Dataset 1')
    plt.scatter(dataset2_embedding[:,0], dataset2_embedding[:,1], color='blue', label='Dataset 2')
    plt.scatter(dataset3_embedding[:,0], dataset3_embedding[:,1], color='green', label='Dataset 3')
    plt.legend()
    plt.savefig(f'{args.model_misc_path}/tsne.png')
    logging.info("Plot saved.")

    
