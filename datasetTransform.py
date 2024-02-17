
import torch 
import torchvision 
from torch.utils.data import Dataset, DataLoader 
import numpy as np
import math 
import torchvision

class WineDataset(Dataset):

       def __init__(self, transform=None):
       # data loading 
              xy = np.loadtxt("wine.csv", delimiter=",",  dtype=np.float32, skiprows=1)     
              self.x = xy[:,1:]
              self.y = xy[:, [0]]
              self.n_samples = xy.shape[0]
              self.transform = transform

       def __getitem__(self, index):
       # dataset indexing 
              sample = self.x[index], self.y[index]
              if self.transform:
                     sample = self.transform(sample)
              return sample


       def __len__(self):
              return self.n_samples

class ToTensor:
       def __call__(self, sample, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
              inputs, targets = sample
              return torch.from_numpy(inputs), torch.from_numpy(targets)


class MulTransform:
       def __init__(self, factor) -> None:
            self.factor = factor
       def __call__(self,sample, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
              inputs, target = sample 
              inputs *= self.factor
              return inputs, target
dataset = WineDataset(transform=ToTensor())


composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])

dataset = WineDataset(transform=composed)


# TEST OUR COMPOSED TRANSFORM 

first_item=  dataset[0]
input, target  = first_item
print(input, target)