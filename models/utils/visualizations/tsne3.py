import torch
import torchvision
from torchvision import transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load and preprocess the three datasets
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image channels
])
dataset1 = torchvision.datasets.ImageFolder('path_to_dataset1', transform=transform)
dataset2 = torchvision.datasets.ImageFolder('path_to_dataset2', transform=transform)
dataset3 = torchvision.datasets.ImageFolder('path_to_dataset3', transform=transform)

# Concatenate the datasets
concat_dataset = torch.utils.data.ConcatDataset([dataset1, dataset2, dataset3])

# Create a data loader
data_loader = torch.utils.data.DataLoader(concat_dataset, batch_size=64, shuffle=False)

# Load a pre-trained CNN model
model = torchvision.models.resnet50(pretrained=True)
model = model.eval()

# Extract features from the datasets
features = []
with torch.no_grad():
    for images, _ in data_loader:
        outputs = model(images)
        features.extend(outputs)

# Convert the features to a tensor
features_tensor = torch.stack(features)

# Compute pairwise feature distances
distances = torch.cdist(features_tensor, features_tensor)

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2)
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
plt.show()
