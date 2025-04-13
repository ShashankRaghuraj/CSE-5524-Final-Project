# CSE-5524-Final-Project

## How to use
Create a python virtual environment and activate it.  

Run `pip install -r requirements.txt`  

Generate the data by running `python3 data/generate_data.py`  

Train the autoencoder by running `python3 pytorch_autoencoder.py -t`. By default, the model will be stored in `autoencoder.pth`. 

To use a model checkpoint for inference only, run `python3 pytorch_autoencoder.py -cm [MODEL]` (omit the train flag). See the main function in pytorch_autoencoder.py for more options, under args.  
