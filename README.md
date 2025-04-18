# CSE-5524-Final-Project

## How to use
Create a python virtual environment and activate it.  

Run `pip install -r requirements.txt`  

Generate the data by running `python3 data/generate_data.py`  

## Autoencoder

Train the autoencoder by running `python3 pytorch_autoencoder.py -t`. By default, the model will be stored in `autoencoder.pth`, which can be changed by setting the save path flag `sp [MODEL_NAME].pth`

To learn from a previous model and continue training, run `python3 pytorch_autoencoder.py -lf [MODEL] -t`

To use a model to generate an animation only, run `python3 pytorch_autoencoder.py -lf [MODEL]` (omit the train flag). See the main function in pytorch_autoencoder.py for more options, under args.  5 random images will be chosen, and an animation will be generated interpolating between those images in encoded space.


## Contrastive Learning

Everything is the same as with the autoencoder, except the python file is pytorch_contrastive, and the model will be stored by default in `contrastive_encoder.pth`.  Also, instead of generating an animation, it will generate a file called `similar_images/similar_images.png`, which will pick 5 random images, and for each image, find the 5 nearest neighbors in encoded space.

It is currently set up to train an encoder that only recognizes color, although we also have the capability to train an encoder that ignores color and cares about the other things. To change to that encoder, set `REPRESENTATION_SIZE` to something larger than 2 (32, for example), and in the main function where the model is created, pass in `ignore_color_transform` instead of `preserve_color_transform`.
