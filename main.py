import torch
from cnn import AudioClassifier
from util import metadata_reader, get_data_loaders
from training import training
from inference import inference

# Settings
num_epochs=2

# Prepare Training
data_folder = 'data/'
metadata_df = metadata_reader(data_folder)
train_dl, val_dl = get_data_loaders(metadata_df, data_folder)

# Create the model and put it on the GPU if available
myModel = AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)

# Check that it is on Cuda
next(myModel.parameters()).device

# Training
training(myModel, train_dl, num_epochs, device)

# Run inference on trained model with the validation set
inference(myModel, val_dl, device)