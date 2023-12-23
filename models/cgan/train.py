from cgan.dataset import FashionMNIST
from cgan.model3 import Discriminator, Generator
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
import os
from PIL import Image

def generator_train_step(batch_size, z_size, class_num, device, discriminator, generator, g_optimizer, criterion):

    # Init gradient
    g_optimizer.zero_grad()

    target = torch.ones((batch_size, 1), requires_grad=True).detach().to(device) # batch_size x 1

    # Building z
    z = torch.randn((batch_size, z_size), requires_grad=True).to(device)

    # Building fake labels
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, class_num, batch_size))).to(device)

    # Generating fake images
    fake_images = generator(z, fake_labels)

    # Disciminating fake images
    validity = discriminator(fake_images, fake_labels)

    # Calculating discrimination loss (fake images)
    g_loss = criterion(validity, target)

    # Backword propagation
    g_loss.backward()

    #  Optimizing generator
    g_optimizer.step()

    return g_loss.data

def discriminator_train_step(batch_size, z_size, class_num, device, discriminator, generator, d_optimizer, criterion, real_images, labels):

    # Init gradient
    d_optimizer.zero_grad()

    target = torch.ones((batch_size, 1), requires_grad=True).detach().to(device) # batch_size x 1

    # Disciminating real images

    # noise = torch.randn(real_images.shape, requires_grad=True).to(device) * 0.1 # small variance
    # real_images = real_images + noise

    print("real_images.shape", real_images.shape)

    real_validity = discriminator(real_images, labels)

    # Calculating discrimination loss (real images)
    real_loss = criterion(real_validity, target)

    # Building z
    z = torch.randn((batch_size, z_size), requires_grad=True).to(device) # noise

    # Building fake labels
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, class_num, batch_size))).to(device)

    # Generating fake images
    fake_images = generator(z, fake_labels)

    print("fake_images.shape", fake_images.shape)

    # Disciminating fake images
    fake_validity = discriminator(fake_images, fake_labels)

    # Calculating discrimination loss (fake images)
    fake_loss = criterion(fake_validity, target)

    # Sum two losses
    d_loss = real_loss + fake_loss

    # Backword propagation
    d_loss.backward()

    # Optimizing discriminator
    d_optimizer.step()

    return d_loss.data

def train():
    # Data
    batch_size = 32  # Batch size

    # Model
    z_size = 100

    # Training
    epochs = 500  # Train epochs
    learning_rate = 2e-4

    data_type = "mnist"
    
    if data_type == "fashion_mnist":

        train_data_path = 'save/fashion-mnist_train.csv' # Path of data
        print('Train data path:', train_data_path)

        img_size = 28 # Image size
        class_list = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, ), std=(0.5, ))
        ])
        dataset = FashionMNIST(train_data_path, img_size, transform=transform)

    else:

        train_data_path = 'save/' # Path of data
        print('Train data path:', train_data_path)

        img_size = 64 # Image size
        dataset = datasets.MNIST(root=train_data_path, download=True,
                                transform=transforms.Compose([
                                    transforms.Resize(img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                ]))


    class_num = 10 #len(class_list)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define generator
    generator = Generator(z_size, img_size, class_num).to(device)
    # Define discriminator
    discriminator = Discriminator(img_size, class_num, batch_size).to(device)

    # Loss function
    criterion = nn.BCELoss()

    # Optimizer
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Create a folder to save the images if it doesn't exist
    output_folder = 'save/output_images'
    os.makedirs(output_folder, exist_ok=True)

    for epoch in range(epochs):

        print('Starting epoch {}...'.format(epoch+1))

        for i, (images, labels) in enumerate(data_loader):

            # Train data
            real_images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            # Set generator train
            generator.train()

            # Train discriminator
            d_loss = discriminator_train_step(batch_size, z_size, class_num, device, discriminator,
                                            generator, d_optimizer, criterion, real_images, labels)

            # Train generator
            g_loss = generator_train_step(batch_size, z_size, class_num, device, discriminator, generator, g_optimizer, criterion)

        # Set generator eval
        generator.eval()

        print('g_loss: {}, d_loss: {}'.format(g_loss, d_loss))

        # Building z
        z = torch.randn((class_num, z_size), requires_grad=True).to(device)

        # Labels 0 ~ 9
        labels = Variable(torch.LongTensor(np.arange(class_num))).to(device)

        # Generating images
        sample_images = generator(z, labels, bs=labels.shape[0]).unsqueeze(1).data.cpu()

        # Show images
        # grid = make_grid(sample_images.squeeze(1), nrow=3, normalize=True).permute(1,2,0).numpy()
        # plt.imshow(grid)
        # plt.show()

        # Save each image separately in the folder
        # for i, image in enumerate(grid):
        for i, image in enumerate(sample_images.squeeze(1)):
            image_path = os.path.join(output_folder, f'image_{i + 1}.png')
            save_image(torch.tensor(image), image_path)

            # Convert the image to a PIL Image before saving
            # pil_image = Image.fromarray((image * 255).astype(np.uint8))
            # pil_image.save(image_path)