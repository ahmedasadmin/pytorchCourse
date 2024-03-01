import  torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import torch
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import io
# hyper parameters
input_size = 784
hidden_size = 100
num_classes = 10
batch_size = 100
learning_rate = 0.001
num_epochs = 3

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        
        return out


FILE = "mnist.pth"
model = NeuralNet(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load(FILE))
model.eval()      

def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
         transforms.ToTensor()
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)


def  get_prediction(image_tensor):
    
    images = image_tensor.reshape(-1, 28*28)
    outputs = model(images)
    _, predicted =  torch.max(outputs.data, 1)
    return predicted 
