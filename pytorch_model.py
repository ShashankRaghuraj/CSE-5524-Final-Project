# This file has old code I don't want to delete.
# It's similar to pytorch_autoencoder but without convolutions

# import torch
# import torch.nn as nn
# import numpy as np
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset, Dataset
# import torchvision.transforms as transforms
# # Import the dataset

# # matplotlib
# import matplotlib.pyplot as plt

# from PIL import Image
# import os
# from tqdm import tqdm


# # Give pytorch access to the GPU

# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
# print(device)



# IMAGE_SIZE = 128

# IMAGE_DIRECTORY = 'data/images'


# DIMS = [200]
# INPUT_DIM = IMAGE_SIZE * IMAGE_SIZE * 3
# REPRESENTATION_SIZE = 20
# NUM_IMAGES = 100


# transform = transforms.Compose([
#     # transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.ToTensor(),
#     # Flatten
#     transforms.Lambda(lambda x: x.view(-1)),
#     # Detatch
#     transforms.Lambda(lambda x: x.detach()),
# ])


# class ImageDataset(Dataset):
#     def __init__(self, image_directory, num_images, transform=None):
#         self.image_directory = image_directory
#         self.num_images = num_images
#         self.transform = transform
#         self.images = self.load_images()

#     def load_images(self):
#         print("Loading images...")
#         images = []
#         for i in tqdm(range(self.num_images), leave=False):
#             img_path = os.path.join(self.image_directory, f'image_{i}.png')
#             if os.path.isfile(img_path):
#                 img = Image.open(img_path)
#                 if self.transform:
#                     img = self.transform(img)
#                 images.append(img)
#         return images

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         return self.images[idx]



# class Autoencoder(nn.Module):
#     def __init__(self, input_dim, representation_dim):
#         super(Autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, DIMS[0]),
#             nn.Sigmoid(),
#             *np.asarray([[nn.Linear(DIMS[i], DIMS[i + 1]), nn.Sigmoid()] for i in range(len(DIMS) - 1)]).flatten().tolist(),
#             nn.Linear(DIMS[-1], representation_dim),
#             nn.Sigmoid(),
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(representation_dim, DIMS[-1]),
#             nn.Sigmoid(),
#             *np.asarray([[nn.Linear(DIMS[i+1], DIMS[i]), nn.Sigmoid()] for i in reversed(range(len(DIMS)- 1))]).flatten().tolist(),
#             nn.Linear(DIMS[0], input_dim),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded
#     def encode(self, x):
#         return self.encoder(x)
#     def decode(self, x):
#         return self.decoder(x)
    
#     def train(self, x):
#         # Forward pass
#         output = self.forward(x)
#         # Compute loss
#         loss = nn.MSELoss()(output, x)
#         # Backward pass
#         loss.backward()
#         return loss.item()
    

# def train_autoencoder(train_model : Autoencoder, data_loader : DataLoader, epochs):
#     for epoch in range(epochs):
#         total_loss = 0
#         for batch in tqdm(data_loader, leave=False):
#             # Move the data to the GPU
#             data = batch.to(device)

#             optimizer.zero_grad()
#             # Batch size
#             # Train the model
#             loss = train_model.train(data)
#             total_loss += loss
#             optimizer.step()
#         print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {total_loss / len(data_loader)}")
#         # Show an example visualization on a random data
#         if epoch % 10 == 0:
#             with torch.no_grad():
#                 example_data = data[0].unsqueeze(0)
#                 output = train_model(example_data)
#                 # Show the original image
#                 plt.subplot(1, 2, 1)
#                 plt.imshow(example_data.view(3, IMAGE_SIZE, IMAGE_SIZE).permute(1, 2, 0).cpu().numpy())
#                 plt.title("Original Image")
#                 # Show the reconstructed image, make sure to unnormalize the image
#                 plt.subplot(1, 2, 2)
#                 plt.imshow(output.view(3, IMAGE_SIZE, IMAGE_SIZE).permute(1, 2, 0).cpu().numpy())
#                 plt.title(f"Epoch {epoch + 1}")
#                 # Save the plot
#                 plt.savefig(f'epoch_{epoch + 1}.png')


# # Create a model
# model = Autoencoder(INPUT_DIM, REPRESENTATION_SIZE).to(device)
# # load the model from autoencoder.pth
# model.load_state_dict(torch.load('autoencoder.pth'))

# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Create a dataset and dataloader
# dataset = ImageDataset(IMAGE_DIRECTORY, NUM_IMAGES, transform=transform)
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
# # Train the model

# print("Training the model...")
# # train_autoencoder(model, data_loader, epochs=100)
# # Save the model
# torch.save(model.state_dict(), 'autoencoder.pth')


# # Pick two random images
# im1 = dataset[np.random.randint(0, len(dataset))]
# im2 = dataset[np.random.randint(0, len(dataset))]
# # Create the images as figures
# with torch.no_grad():
#     im1_s = im1.view(3, IMAGE_SIZE, IMAGE_SIZE).permute(1, 2, 0).cpu().numpy()
#     im2_s = im2.view(3, IMAGE_SIZE, IMAGE_SIZE).permute(1, 2, 0).cpu().numpy()
# # Create a figure
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(im1_s)
# plt.axis('off')
# plt.title("Image 1")
# plt.subplot(1, 2, 2)
# plt.imshow(im2_s)
# plt.axis('off')
# plt.title("Image 2")
# plt.savefig('images.png')
# plt.close()
# im1 = im1.to(device)
# im2 = im2.to(device)


# def create_animation(model, im1, im2, num_frames=30):
#     # Create a list of frames
#     # frames = []
#     with torch.no_grad():
#         # Encode the images
#         encoded_im1 = model.encode(im1)
#         encoded_im2 = model.encode(im2)
#         for i in range(num_frames):
#             # Interpolate between the two images
#             alpha = i / (num_frames - 1)
#             interpolated = (1 - alpha) * encoded_im1 + alpha * encoded_im2
#             # Decode the interpolated image
#             decoded = model.decode(interpolated)
#             # Reshape the decoded image
#             decoded = transforms.ToPILImage()(decoded.view(3, IMAGE_SIZE, IMAGE_SIZE).permute(1, 2, 0).cpu().numpy())
#             # Save the frame
#             decoded.save(f'frame_{i}.png')

#     # return frames

# create_animation(model, im1, im2)