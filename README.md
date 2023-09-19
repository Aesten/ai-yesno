# Yes/No voice recognition AI

## Project specifications

- PyTorch is the library used for this AI (install using `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118` or check on the official website to install accordingly to your system)
- You also need soundfile and pandas (`pip install soundfile pandas`) 
- The CNN has 4 convolutional layers which may be a bit over the top for this application, but could be reused for more complex audio classification
- The code can be quickly modified for multiclass classification (>2), simply change the last layer of the CNN
- The audio preprocessing parameters are all "hard-coded" in the SoundDS class of `util.py`, but you can modify it to suit your needs

## The files

- `util.py` contains classes and methods/functions to prepare the dataloader. Some functions here are also responsible to resize audios, rechannel or resample them to have a standardized input for the NN.
- `cnn.py` contains the the code for the CNN.
- `training.py` contains the training process of the CNN.
- `inference.py` contains the code for the inference (test of accuracy on a subset of data) of the AI.
- `main.py` is the main code to launch the training. Execute this file using `python main.py` to launch the training.
- `cuda.py` is a simple script that outputs a boolean telling you if you have cuda available/enabled on your device.

## Results

Here is the console output during a training on 10 epochs:

> Preparing data  
> Device:cuda:0  
> Starting training  
> Epoch: 0, Loss: 0.68, Accuracy: 0.57  
> Epoch: 1, Loss: 0.60, Accuracy: 0.70  
> Epoch: 2, Loss: 0.37, Accuracy: 0.86  
> Epoch: 3, Loss: 0.25, Accuracy: 0.90  
> Epoch: 4, Loss: 0.21, Accuracy: 0.92  
> Epoch: 5, Loss: 0.18, Accuracy: 0.93  
> Epoch: 6, Loss: 0.16, Accuracy: 0.94  
> Epoch: 7, Loss: 0.15, Accuracy: 0.94  
> Epoch: 8, Loss: 0.14, Accuracy: 0.95  
> Epoch: 9, Loss: 0.14, Accuracy: 0.94  
> Finished Training  
> Testing the model  
> Accuracy: 0.95, Total items: 1597

