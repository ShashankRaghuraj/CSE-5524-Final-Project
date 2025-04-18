import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision.transforms as transforms
# Import the dataset


import argparse
from PIL import Image
import os
from tqdm import tqdm
import random
import colorsys


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
TEMPERATURE=0.5

IMAGE_DIRECTORY = 'data/images'


INPUT_DIM = IMAGE_SIZE * IMAGE_SIZE * 3
REPRESENTATION_SIZE = 2 # Should increase to something like 32 if using 
NUM_EPOCHS= 10

# m value for the triplet loss
MARGIN = 0.3

# Will load in `image_0.png` through `imagine_{NUM_IMAGES-1}.png`
# If you are just trying to make an animation, you only need to load a few images
NUM_IMAGES = 50_000


class ImageDataset(Dataset):
    # transform runs on everything, invariant_transform is the thing we want to train our encoder to be invariant to
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


class ContrastiveEncoder(nn.Module):
    def __init__(self, input_dim, representation_dim, invariant_transform):
        super(ContrastiveEncoder, self).__init__()
        self.invariant_transform = invariant_transform
        self.activation_fn = nn.ReLU
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # 32x128x128 => 64x64x64
            self.activation_fn(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), # 64x64x64 => 64x32x32
            self.activation_fn(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), # 64x32x32 => 64x16x16
            self.activation_fn(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), # 64x16x16 => 128x8x8
            self.activation_fn(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), # 128x8x8 => 128x4x4
            self.activation_fn(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), # 128x4x4 => 128x2x2
            self.activation_fn(),
            nn.Conv2d(32,REPRESENTATION_SIZE, kernel_size=2, padding = 0), # 128x2x2 => 2x1x1
            nn.Flatten(), # 2x1x1 => 2
            nn.Tanh()
        )
        self.triplet_loss = nn.TripletMarginLoss(margin=MARGIN)

    def contrast_loss(self, output, transform_output):
        # Calculate the triplet loss
        # Generate x- by rotating the output (comparing every element to some other random element)
        rand_roll = np.random.randint(1, len(output))
        output_minus = torch.roll(output, shifts=-rand_roll, dims=0)
        return self.triplet_loss(output, transform_output, output_minus)



    def forward(self, x):
        return self.encoder(x)
    
    def train(self, x):
        transform_data = self.invariant_transform(x)
        # If any of the transformed images are all 0, remove them
        mask = torch.any(transform_data != 0, dim=(1, 2, 3))
        transform_data = transform_data[mask]
        x = x[mask]
        output = self.forward(x)
        transform_output = self.forward(transform_data)
        loss = self.contrast_loss(output, transform_output)
        # Backward pass
        loss.backward()
        return loss.item()

def train_encoder(train_model : ContrastiveEncoder, data_loader : DataLoader, epochs):
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
        # Get a small sample and calculate jus tthe positive loss
        with torch.no_grad():
            sample = next(iter(data_loader))
            sample = sample.to(device)
            output = train_model.forward(sample)
            transform_output = train_model.forward(train_model.invariant_transform(sample))
            # Calculate the mse loss between the two
            mse_loss = nn.MSELoss()
            loss = mse_loss(output, transform_output)
            # Print the average loss for the sample as positive loss
            print(f"Sample Positive Loss: {loss.item()}")
            # Also, get the average difference between two outputs
            rolled_output = torch.roll(output, shifts=-1, dims=0)
            loss = mse_loss(output, rolled_output)
            print(f"Sample Negative Loss: {loss.item()}")
            # Generate a graph, should only use if training the model to recognize color only and the representation size is 2
            if REPRESENTATION_SIZE == 2:
                generate_graph(train_model, data_loader)



def find_most_similar(train_model : ContrastiveEncoder, data_loader : DataLoader, similar_to_images, num_images=5):
    # find the n most similar images to image
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(similar_to_images), num_images + 1, figsize=(num_images * 2 + 2, len(similar_to_images) * 2))
    with torch.no_grad():
        for i, similar_to_image in enumerate(similar_to_images):
            image = similar_to_image.to(device)
            image = image.unsqueeze(0)
            image = train_model.forward(image)
            image = image.squeeze(0)
            image = image.cpu().numpy()
            distances = []
            for batch in tqdm(data_loader, leave=False):
                batch = batch.to(device)
                batch = train_model.forward(batch)
                batch = batch.squeeze(0)
                batch = batch.cpu().numpy()
                # Calculate distance between image and each element in the batch
                distance = np.linalg.norm(image - batch, axis=1)
                for dist in distance:
                    distances.append(dist)

            distances = np.array(distances)
            # Get the indices of the most similar images
            indices = np.argsort(distances)[1:num_images+1]
            images = []
            for j in indices:
                images.append(data_loader.dataset.images[j])
            # Create a plot with all of these images
            
            # show the original image
            axs[i,0].imshow(similar_to_image.permute(1, 2, 0).cpu().numpy())
            axs[i,0].axis('off')
            for j in range(num_images):
                axs[i,j + 1].imshow(images[j].permute(1, 2, 0).cpu().numpy())
                axs[i,j + 1].axis('off')
            # Save the plot
            if not os.path.exists("similar_images"):
                os.makedirs("similar_images")
            plt.savefig(f'similar_images/similar_images.png')
    


def generate_graph(model, data_loader, num_batches=4):
    # Graph each image's encoding on a 2D graph, and color them by their actual RGB color
    import matplotlib.pyplot as plt

    points = []
    data_loader_iter = iter(data_loader)
    for i in range(num_batches):
        batch = next(data_loader_iter)
        batch = batch.to(device)
        with torch.no_grad():
            output = model.forward(batch)
            output = output.cpu().numpy()
            for i, (x, y) in enumerate(output):
                # Get the color of the image
                img = batch[i].cpu().numpy()
                # The color is the maximum r, maximum g, maximum b value
                color = (img[0].max(), img[1].max(), img[2].max())
                points.append((x,y, color))
    # Create two 2d scatter plots of the points

    fig, ax = plt.subplots()
    ax.set_title('2D Scatter Plot of Encodings')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    for x, y, color in points:
        ax.scatter(x, y, color=color)
    # Save the plot
    if not os.path.exists("encoder_output"):
        os.makedirs("encoder_output")
    plt.savefig(f'encoder_output/encoder_output.png')


def randomize_color(image):
    # Transformation, replaces the color of the image with a random color
    h = random.uniform(0, 1)
    s = random.uniform(0.5, 1.0)
    l = random.uniform(0.3, 0.7)
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    # Turn every nonzero value to 1
    image = image.clone()
    image[image != 0] = 1
    return image * torch.tensor([r, g, b]).view(3, 1, 1).to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-lf', "--load_from", default=None, help="Load model from previously saved .pth file")
    parser.add_argument('-t', "--train", action='store_true', help="Whether to do model training. If inference is just needed, model path should also be passed in.")
    parser.add_argument('-sp', "--save_path", default="contrastive_encoder.pth", help="Save path of the trained model. Default is contrastive_encoder.pth")
    args = parser.parse_args()
    
    # Sample invariant transforms
    preserve_color_transform = transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(.08, .9), ratio=(0.5, 2))
    ignore_color_transform = transforms.Compose([
        transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(.9, 1), ratio=(1, 1)),
        # transforms.ColorJitter(hue=0.5, saturation=0.2)
        transforms.Lambda(randomize_color),
    ])

    # Create a model
    model = ContrastiveEncoder(INPUT_DIM, REPRESENTATION_SIZE, preserve_color_transform).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0002)

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
        train_encoder(model, data_loader, epochs=NUM_EPOCHS)
        torch.save(model.state_dict(), args.save_path)
    else:
        assert args.load_from, "No model checkpoint was passed in"
    
    # Now, find the most similar images to a random image
    random_images = [random.choice(dataset.images) for _ in range(5)]
    noshuffle_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    find_most_similar(model, noshuffle_loader, random_images)
