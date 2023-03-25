import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image

from datautils.target_dataset import get_target_pretrain_ds
from models.utils.training_type_enum import TrainingType

class FeatureSim():
    def __init__(self, args) -> None:
        self.args = args

    def get_loader(self):
        # Define a transform to preprocess the data
        transform = transforms.Compose([
            transforms.Resize(28), 
            transforms.ToTensor(), 
            transforms.Normalize((0.5,), (0.5,))
        ])

        # # Load the two datasets
        dataset1 = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
        # # dataset1 = Dataset(dataset1)

        dataset2 = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
        # # dataset2 = Dataset(dataset2)

        # ds_1 = get_target_pretrain_ds(self.args, training_type=TrainingType.BASE_PRETRAIN)
        # dataset1 = ds_1.get_dataset(transform, is_tsne=True)

        # ds_2 = get_target_pretrain_ds(self.args, training_type=TrainingType.ACTIVE_LEARNING)
        # dataset2 = ds_2.get_dataset(transform, is_tsne=True)

        # Combine the datasets into a single tensor
        combined_data = torch.utils.data.ConcatDataset([dataset1, dataset2])

        # Create a dataloader to load the data in batches
        dataloader = torch.utils.data.DataLoader(combined_data, batch_size=1000, shuffle=True)

        return dataloader

    def compute_similarity(self):
        # Load a pre-trained model (e.g. a convolutional neural network)
        model = torchvision.models.resnet18(pretrained=True)

        # Remove the last layer of the model to obtain feature vectors
        model = torch.nn.Sequential(*list(model.children())[:-1])

        dataloader = self.get_loader()

        # Extract the features from the data using the pre-trained model
        features, labels = self.extract_features(model, dataloader)

        nsamples, c, nx, ny = features.shape
        features = features.reshape((nsamples,c*nx*ny))

        # Apply t-SNE to the feature vectors to reduce the dimensionality of the data
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embeddings = tsne.fit_transform(features)

        # Plot the t-SNE embeddings, coloring the data points based on their original dataset label
        plt.scatter(embeddings[labels==0, 0], embeddings[labels==0, 1], color='#7089b8', label='MNIST')
        plt.scatter(embeddings[labels==1, 0], embeddings[labels==1, 1], color='#f69a2a', label='FashionMNIST')
        plt.legend()
        plt.savefig('tsne.png')
        # plt.show()

    # Define a function to extract features from the data using a pre-trained model
    def extract_features(self, model, dataloader):
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
