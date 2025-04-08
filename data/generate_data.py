import random
import colorsys
from PIL import Image, ImageDraw
import math


NUM_IMAGES = 50_000
IMAGE_SIZE = 128
SQRT3_2 = math.sqrt(3) / 2


def pick_random_color():
    # Random color that's not too dark or too light
    h = random.uniform(0, 1)
    s = random.uniform(0.5, 1.0)
    l = random.uniform(0.3, 0.7)
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return int(r * 255), int(g * 255), int(b * 255)

def generate_image():
    r, g, b = pick_random_color()
    # Chose a random shape
    shape = random.choice(['circle', 'square', 'triangle'])
    # Choose a random size
    size = random.randint(10, 100)
    # Choose a random position
    x = random.randint(0, IMAGE_SIZE - size)
    y = random.randint(0, IMAGE_SIZE - size)
    # Choose a random rotation
    angle = random.randint(0, 360)
    # Create a blank image
    image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    # Draw the shape
    if shape == 'circle':
        draw.ellipse((x, y, x + size, y + size), fill=(r, g, b))
    elif shape == 'square':
        draw.rectangle((x, y, x + size, y + size), fill=(r, g, b))
    elif shape == 'triangle':
        draw.polygon([(x, y), (x + size, y), (x + size // 2, y + int(size*SQRT3_2))], fill=(r, g, b))
    # Rotate the image
    image = image.rotate(angle, expand=True)
    # Resize the image
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    return image


if __name__ == '__main__':
    # Create a directory to save the images
    import os
    if not os.path.exists('data/images'):
        os.makedirs('data/images')
    # Generate a lot of images
    for i in range(NUM_IMAGES):
        image = generate_image()
        image.save(f'data/images/image_{i}.png')


