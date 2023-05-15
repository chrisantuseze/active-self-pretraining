import torch
import torchvision
import torchvision.models as models

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

# Define a function to extract features from the data using a pre-trained model
def extract_features(model, dataset):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=512, 
        shuffle=True, num_workers=4,
        pin_memory=True,)

    logging.info("Extracting features...")

    features = []
    with torch.no_grad():
        for images, target in dataloader:
            
            # Move the data to the GPU if available
            if torch.cuda.is_available():
                images = images.cuda()
            
            # Extract the features using the pre-trained model
            output = model(images)
            features.append(output.cpu())
    
    return features

def tsne_similarity(args):
    # Load the ResNet-18 model
    model = models.resnet18(pretrained=True)
    model.eval()

    # Remove the last fully connected layer
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model = model.to(args.device)

    transform = Transforms(args.target_image_size)

    # Load images from the three datasets and extract their features
    args.target_dataset = 12
    ds_1 = get_target_pretrain_ds(args, training_type=TrainingType.ACTIVE_LEARNING)
    dataset1 = ds_1.get_dataset(transform, is_tsne=True)

    ds = f"generated_{get_dataset_enum(args.target_dataset)}"
    img_path = get_images_pathlist(f'{args.dataset_dir}/{ds}', with_train=True)
    path_loss_list = [PathLoss(path, 0) for path in img_path]
    dataset2 = PretextDataset(args, path_loss_list, transform, False)

    args.target_dataset = 13
    ds_3 = get_target_pretrain_ds(args, training_type=TrainingType.ACTIVE_LEARNING)
    dataset3 = ds_3.get_dataset(transform, is_tsne=True)

    # Concatenate the datasets
    concat_dataset = torch.utils.data.ConcatDataset([dataset1, dataset2, dataset3])

    # Create a data loader
    data_loader = torch.utils.data.DataLoader(concat_dataset, batch_size=512, shuffle=False)


    logging.info("Extracting features...")
    # Extract features from the datasets
    features = []
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(args.device)
            outputs = model(images)
            features.extend(outputs)

    # Convert the features to a tensor
    features_tensor = torch.stack(features)

    # Compute pairwise feature distances
    distances = torch.cdist(features_tensor, features_tensor)

    logging.info("Generating TSNE embeddings...")

    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)

    distances = distances.cpu()
    nsamples, nx, ny = distances.shape
    distances = distances.reshape((nsamples, nx*ny))

    embeddings = tsne.fit_transform(distances)

    # Plot the t-SNE embeddings
    num_samples_dataset1 = len(dataset1)
    num_samples_dataset2 = len(dataset2)
    num_samples_dataset3 = len(dataset3)

    plt.scatter(embeddings[:num_samples_dataset1, 0], embeddings[:num_samples_dataset1, 1], c='red', label='Dataset 1')
    plt.scatter(embeddings[num_samples_dataset1:num_samples_dataset1+num_samples_dataset2, 0], 
                embeddings[num_samples_dataset1:num_samples_dataset1+num_samples_dataset2, 1], c='blue', label='Dataset 2')
    plt.scatter(embeddings[num_samples_dataset1+num_samples_dataset2:, 0], 
                embeddings[num_samples_dataset1+num_samples_dataset2:, 1], c='green', label='Dataset 3')
    plt.legend()
    plt.savefig(f'{args.model_misc_path}/tsne.png')
    logging.info("Plot saved.")
