import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision.transforms as transforms
# Import the dataset

# matplotlib
import matplotlib.pyplot as plt

import argparse
from PIL import Image
import os
from tqdm import tqdm


# Give pytorch access to the GPU

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device " + str(device))


IMAGE_SIZE = 128
BATCH_SIZE=32

IMAGE_DIRECTORY = 'data/images'


INPUT_DIM = IMAGE_SIZE * IMAGE_SIZE * 3
REPRESENTATION_SIZE = 128
NUM_EPOCHS= 10

# Will load in `image_0.png` through `imagine_{NUM_IMAGES-1}.png`
# If you are just trying to make an animation, you only need to load a few images
NUM_IMAGES = 50_000

# Show examples of the autoencoder output while training
SHOW_EXAMPLES = True


class ImageDataset(Dataset):
    def __init__(self, image_directory, num_images, transform=None):
        self.image_directory = image_directory
        self.num_images = num_images
        self.transform = transform
        self.images = self.load_images()

    def load_images(self):
        print("Loading images...")
        images = []
        for i in tqdm(range(self.num_images), leave=False):
            img_path = os.path.join(self.image_directory, f'image_{i}.png')
            if os.path.isfile(img_path):
                img = Image.open(img_path)
                if self.transform:
                    img = self.transform(img)
                images.append(img)
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Original image, transform image
        return self.images[idx]



class Autoencoder(nn.Module):
    def __init__(self, input_dim, representation_dim):
        self.activation_fn = nn.ReLU
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # 3x128x128 => 32x128x128
            self.activation_fn(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 32x128x128 => 64x64x64
            self.activation_fn(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # 64x64x64 => 64x32x32
            self.activation_fn(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # 64x32x32 => 64x16x16
            self.activation_fn(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 64x16x16 => 128x8x8
            self.activation_fn(),
            nn.Flatten(), # 128x8x8 => 128*8*8
            nn.Linear(128 * 8 * 8, representation_dim), # 128*8*8 => ??
            self.activation_fn()
        )
        self.decoder = nn.Sequential(
            nn.Linear(representation_dim, 128 * 8 * 8), # ?? => 128*8*8
            self.activation_fn(),
            nn.Unflatten(1, (128, 8, 8)), # 128*8*8 => 128x8x8
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1, padding=1), # 128x8x8 => 64x16x16
            self.activation_fn(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # 64x16x16 => 64x16x16
            self.activation_fn(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, output_padding=1, padding=1), # 64x16x16 => 64x32x32
            self.activation_fn(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # 64x32x32 => 64x32x32
            self.activation_fn(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, output_padding=1, padding=1), # 64x32x32 => 32x64x64
            self.activation_fn(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), # 32x64x64 => 32x64x64
            self.activation_fn(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, output_padding=1, padding=1), # 32x64x64 => 3x128x128
            self.activation_fn(),
            nn.Conv2d(3, 3, kernel_size=3, padding=1), # 3x128x128 => 3x128x128
            nn.ReLU(),
            nn.Tanh() # Relu + tanh to force between 0 and 1 without making it too hard to get a 0 value
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)
    def decode(self, x):
        return self.decoder(x)
    
    def train(self, x):
        # Compute loss
        output = self.forward(x)
        loss = self.loss_fn(output, x)
        # Backward pass
        loss.backward()
        return loss.item()


def train_autoencoder(train_model : Autoencoder, data_loader : DataLoader, epochs):
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(data_loader, leave=False):
            # Move the data to the GPU
            data = batch.to(device)

            optimizer.zero_grad()
            # Train the model
            loss = train_model.train(data)
            total_loss += loss
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {total_loss / len(data_loader)}")
        # Show an example visualization on a random data
        if SHOW_EXAMPLES:
            with torch.no_grad():
                example_data = data[0].unsqueeze(0)
                output = train_model(example_data)
                # Show the original image
                plt.subplot(1, 2, 1)
                plt.imshow(example_data.view(3, IMAGE_SIZE, IMAGE_SIZE).permute(1, 2, 0).cpu().numpy())
                plt.title("Original Image")
                # Show the reconstructed image, make sure to unnormalize the image
                plt.subplot(1, 2, 2)
                plt.imshow(output.view(3, IMAGE_SIZE, IMAGE_SIZE).permute(1, 2, 0).cpu().numpy())
                plt.title(f"Epoch {epoch + 1}")
                # Save the plot
                if not os.path.exists("decoder_output"):
                    os.makedirs("decoder_output")
                plt.savefig(f'decoder_output/epoch_{epoch + 1}.png')


# Function to create an animation from a list of images (keyframes) using the autoencoder
def create_animation(model, images, num_frames=30, output_file='animation.gif'):
    # Create a list of frames
    import imageio
    frames = []
    with torch.no_grad():
        # Encode the images
        encoded_ims = [model.encode(im.unsqueeze(0)) for im in images]
        for i in range(len(encoded_ims)):
            encoded_im1 = encoded_ims[i]
            encoded_im2 = encoded_ims[(i + 1) % len(encoded_ims)]
            # Create a list of frames for the animation
            # frames.append(images[i].cpu())
            for j in range(num_frames):
                # Interpolate between the two images
                alpha = j / (num_frames - 1)
                interpolated = (1 - alpha) * encoded_im1 + alpha * encoded_im2
                # Decode the interpolated image
                decoded = model.decode(interpolated)
                # Reshape the decoded image
                decoded = transforms.ToPILImage()(decoded.view(3, IMAGE_SIZE, IMAGE_SIZE).permute(1, 2, 0).cpu().numpy())
                # Save the frame
                frames.append(decoded)
            
    imageio.mimsave(output_file, frames, fps=20, loop=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-lf', "--load_from", default=None, help="Load model from previously saved .pth file")
    parser.add_argument('-t', "--train", action='store_true', help="Whether to do model training. If inference is just needed, model path should also be passed in.")
    parser.add_argument('-sp', "--save_path", default="autoencoder.pth", help="Save path of the trained model. Default is autoencoder.pth")
    args = parser.parse_args()
    
    # Create a model
    model = Autoencoder(INPUT_DIM, REPRESENTATION_SIZE).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create a dataset and dataloader
    dataset = ImageDataset(IMAGE_DIRECTORY, NUM_IMAGES, transform=transforms.ToTensor())
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    ##### LOAD IN PREEXISTING MODEL IF YOU HAVE ONE
    if args.load_from:
        print("Loading model from checkpoint...")
        model.load_state_dict(torch.load(args.load_from))

    ##### CODE TO ACTUALLY TRAIN THE MODEL
    if args.train:
        print("Training the model...")
        train_autoencoder(model, data_loader, epochs=NUM_EPOCHS)
        torch.save(model.state_dict(), args.save_path)
    else:
        assert args.load_from, "No model was passed in"

    #### CODE TO GENERATE AN ANIMATION OUT OF SOME RANDOM IMAGES
    # Pick n random images
    
    n = 5
    images = [dataset[np.random.randint(0, len(dataset))].to(device) for _ in range(n)]

    create_animation(model, images)
