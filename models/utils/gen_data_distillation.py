import glob
import torch
import torchvision
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from datautils.dataset_enum import get_dataset_enum

from models.gan5.operation import ImageFolder
import utils.logger as logging

def prepare_gen_dataset(args):
    ds_dir1 = f"{args.dataset_dir}/{args.base_dataset}"
    ds_dir2 = get_dataset_enum(args.target_dataset)

    # Load the images into PyTorch tensors
    data_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    gen_dataset = ImageFolder(ds_dir1, transform=data_transforms, distillation=True)
    target = ImageFolder(ds_dir2, transform=data_transforms)

    combined_data = torch.utils.data.ConcatDataset([gen_dataset, target])
    dataloader = torch.utils.data.DataLoader(combined_data, batch_size=32, shuffle=False)

    # Apply a pre-trained CNN to extract features from the images
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()
    features = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            outputs = model(inputs)
            features.append(outputs)
    features = torch.cat(features, dim=0)

    # Cluster the images using k-means
    kmeans = KMeans(n_clusters=1, random_state=0).fit(features.numpy())#5

    # Compute the distance between each image and its cluster centroid
    distances = cdist(features.numpy(), kmeans.cluster_centers_, 'euclidean')

    # Select the images that are far away from the centroid
    threshold = 10  # choose a suitable threshold
    far_away_indices = []
    for i in range(len(combined_data)):
        logging.info(f'The distance of the image from the centroid is: {distances[i]}')
        if any(distances[i] > threshold):
            far_away_indices.append(i)

    # Retrieve the far away images from the dataset
    far_away_images = [combined_data[i][0] for i in far_away_indices]

    new_ds = []
    for img in far_away_images:
        if img in gen_dataset:
            new_ds.append(img)

    #save the images to a folder
