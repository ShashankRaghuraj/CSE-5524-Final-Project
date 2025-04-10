import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision.transforms as transforms
# Import the dataset

# matplotlib
import matplotlib.pyplot as plt

from PIL import Image
import os
from tqdm import tqdm


# Give pytorch access to the GPU

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)



IMAGE_SIZE = 128

IMAGE_DIRECTORY = 'data/images'


INPUT_DIM = IMAGE_SIZE * IMAGE_SIZE * 3
REPRESENTATION_SIZE = 128
NUM_EPOCHS= 10

# Will load in `image_0.png` through `imagine_{NUM_IMAGES-1}.png`
NUM_IMAGES = 100


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
            nn.Linear(128 * 8 * 8, REPRESENTATION_SIZE), # 128*8*8 => ??
            self.activation_fn()
        )
        self.decoder = nn.Sequential(
            nn.Linear(REPRESENTATION_SIZE, 128 * 8 * 8), # ?? => 128*8*8
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
        # Forward pass
        output = self.forward(x)
        # Compute loss
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
            # Batch size
            # Train the model
            loss = train_model.train(data)
            total_loss += loss
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {total_loss / len(data_loader)}")
        # Show an example visualization on a random data
        if epoch % 1 == 0:
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
                plt.savefig(f'epoch_{epoch + 1}.png')


# Function to create an animation from two images using the autoencoder
def create_animation(model, im1, im2, num_frames=30):
    # Create a list of frames
    import imageio
    frames = []
    with torch.no_grad():
        # Encode the images
        encoded_im1 = model.encode(im1.unsqueeze(0))
        encoded_im2 = model.encode(im2.unsqueeze(0))
        for i in range(num_frames):
            # Interpolate between the two images
            alpha = i / (num_frames - 1)
            interpolated = (1 - alpha) * encoded_im1 + alpha * encoded_im2
            # Decode the interpolated image
            decoded = model.decode(interpolated)
            # Reshape the decoded image
            decoded = transforms.ToPILImage()(decoded.view(3, IMAGE_SIZE, IMAGE_SIZE).permute(1, 2, 0).cpu().numpy())
            # Save the frame
            frames.append(decoded)
    imageio.mimsave('animation.gif', [frames[0]] * 3 + frames + [frames[-1]] * 3 + list(reversed(frames)), fps=20, loop=0)


# Create a model
model = Autoencoder(INPUT_DIM, REPRESENTATION_SIZE).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create a dataset and dataloader
dataset = ImageDataset(IMAGE_DIRECTORY, NUM_IMAGES, transform=transforms.ToTensor())
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
# Train the model

model.load_state_dict(torch.load('autoencoder.pth'))

# print("Training the model...")
# train_autoencoder(model, data_loader, epochs=NUM_EPOCHS)

# # Save the model
# torch.save(model.state_dict(), 'autoencoder.pth')


# Pick two random images
im1 = dataset[np.random.randint(0, len(dataset))]
im2 = dataset[np.random.randint(0, len(dataset))]

im1 = im1.to(device)
im2 = im2.to(device)

create_animation(model, im1, im2, num_frames=30)
