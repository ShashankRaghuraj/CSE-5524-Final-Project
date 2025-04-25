from pytorch_autoencoder import Autoencoder, IMAGE_SIZE, device, INPUT_DIM, REPRESENTATION_SIZE, create_animation
from pytorch_contrastive import ContrastiveEncoder, find_most_similar
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import os

NUM_IMAGES = 5_000

# Create a custom dataset class
class ImageDataset(Dataset):
    def __init__(self):
        self.num_images = NUM_IMAGES
        self.images = self.load_images()

    def load_images(self):
        print("Loading images...")
        # Load images from 5kimages.npz
        images = np.load('data/5kimages.npz')['images']
        images = np.transpose(images, (0, 3, 1, 2))
        images = torch.from_numpy(images).float() / 255.0
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


def generate_autoencoder_output(input_image):
    autoencoder = Autoencoder(INPUT_DIM, REPRESENTATION_SIZE)
    autoencoder.load_state_dict(torch.load('autoencoder.pth'))
    autoencoder = autoencoder.to(device)
    # Pick a random image from the dataset
    
    # Encode the image
    with torch.no_grad():
        output = autoencoder.forward(input_image)
    
    # Show the original image
    plt.subplot(1, 2, 1)
    plt.imshow(input_image.view(3, IMAGE_SIZE, IMAGE_SIZE).permute(1, 2, 0).cpu().numpy())
    plt.title("Original Image")
    # Show the reconstructed image
    plt.subplot(1, 2, 2)
    plt.imshow(output.view(3, IMAGE_SIZE, IMAGE_SIZE).permute(1, 2, 0).cpu().numpy())
    plt.title(f"Autoencoder Output")
    # Save the plot
    plt.savefig(f'output_images/autoencoder_output.png')
    print("Autoencoder output saved to output_images/autoencoder_output.png")


if __name__ == '__main__':
    # Load the dataset
    dataset = ImageDataset()
    
    # Get the inputted command
    import argparse
    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('command', type=str, help='Command to run, must be one of "test_autoencoder", "create_animation", "color_knn", "colorless_knn"')
    args = parser.parse_args()
    command = args.command
    if not os.path.exists("output_images"):
        os.makedirs("output_images")
    print(f"Running command: {command}")
    if command == 'test_autoencoder':
        # Load the autoencoder
        idx = np.random.randint(0, len(dataset))
        image = dataset[idx].unsqueeze(0)
        generate_autoencoder_output(image.to(device))
    elif command == 'create_animation':
        # Create an animation from the autoencoder
        model = Autoencoder(INPUT_DIM, REPRESENTATION_SIZE)
        model.load_state_dict(torch.load('autoencoder.pth'))
        model = model.to(device)
        # Pick 5 random images from the dataset
        indices = np.random.choice(len(dataset), 5, replace=False)
        images = [dataset[i].to(device) for i in indices]
        create_animation(model, images, output_file='output_images/animation.gif')
        print("Animation saved to output_images/animation.gif")
    elif command == 'color_knn' or command == 'colorless_knn':
        if command == 'colorless_knn':
            model = ContrastiveEncoder(INPUT_DIM, 32, None)
            model.load_state_dict(torch.load('colorless_encoder.pth'))
            output_filename = 'colorless_similar_images.png'
        else:
            model = ContrastiveEncoder(INPUT_DIM, 2, None)
            model.load_state_dict(torch.load('color_contrastive_encoder.pth'))
            output_filename = 'color_similar_images.png'
        model = model.to(device)

        # Pick 5 random images from the dataset
        indices = np.random.choice(len(dataset), 5, replace=False)
        images = [dataset[i].to(device) for i in indices]
        # Create a dataloader
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        # Finding the most similar images
        find_most_similar(model, data_loader, images, output_file=f'output_images/{output_filename}')
        print(f"K Nearest Neighbors output saved to output_images/{output_filename}")
    else:
        print(f"Unknown command: {command}")
        print("Available commands: test_autoencoder, create_animation, color_knn, colorless_knn")
