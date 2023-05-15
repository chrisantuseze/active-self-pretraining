import torch
import torchvision
from torchvision import transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import utils.logger as logging

from datautils.dataset_enum import get_dataset_enum
from datautils.path_loss import PathLoss

from datautils.target_dataset import get_target_pretrain_ds
from models.active_learning.pretext_dataloader import PretextDataset
from models.utils.commons import get_images_pathlist
from models.utils.training_type_enum import TrainingType
from models.utils.transformations import Transforms

# Function to extract features from a dataset
def extract_features(args, model, data_loader):
    logging.info("Extracting features...")
    features = []
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(args.device)
            outputs = model(images)
            features.extend(outputs)
    return torch.stack(features)

def tsne_similarity(args):
    transform = Transforms(args.target_image_size)
    # Load images from the three datasets and extract their features
    args.target_dataset = 12
    ds_1 = get_target_pretrain_ds(args, training_type=TrainingType.ACTIVE_LEARNING)
    dataset1 = ds_1.get_dataset(transform, is_tsne=True)

    ds = f"generated_{get_dataset_enum(args.target_dataset)}"
    img_path = get_images_pathlist(f'{args.dataset_dir}/{ds}', with_train=True)
    path_loss_list = [PathLoss(path, 0) for path in img_path]
    dataset2 = PretextDataset(args, path_loss_list, transform, False)

    # Create data loaders for each dataset
    data_loader1 = torch.utils.data.DataLoader(dataset1, batch_size=512, shuffle=False)
    data_loader2 = torch.utils.data.DataLoader(dataset2, batch_size=512, shuffle=False)

    # Load a pre-trained CNN model
    model = torchvision.models.resnet50(pretrained=True).to(args.device)
    model = model.eval()

    # Extract features from each dataset
    features1 = extract_features(args, model, data_loader1)
    features2 = extract_features(args, model, data_loader2)

    logging.info("Generating TSNE embeddings...")

    # Compute pairwise feature distances
    distances = torch.cdist(features1, features2)

    distances = distances.cpu()

    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    embeddings = tsne.fit_transform(distances)

    # Separate the embedded data into the original datasets
    embedded_data1 = embeddings[:len(dataset1)]
    embedded_data2 = embeddings[len(dataset1):]

    # Create a scatter plot
    plt.scatter(embedded_data1[:, 0], embedded_data1[:, 1], c='#ed9a68', label='Data 1')
    plt.scatter(embedded_data2[:, 0], embedded_data2[:, 1], c='#698e77', label='Data 2')


    # plt.scatter(embeddings[:num_samples_dataset1, 0], embeddings[:num_samples_dataset1, 1], c='#ed9a68', label='Artistic')
    
    # plt.scatter(embeddings[num_samples_dataset1:, 0], 
    #             embeddings[num_samples_dataset1:, 1], c='#698e77', label='Intermediate')

    # Plot the t-SNE embeddings
    # plt.scatter(embeddings[:, 0], embeddings[:, 1])
    plt.legend()
    plt.savefig(f'{args.model_misc_path}/tsne.png')
    logging.info("Plot saved.")
