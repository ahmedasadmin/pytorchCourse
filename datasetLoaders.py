
import torch 
import torchvision 
from torch.utils.data import Dataset, DataLoader 
import numpy as np
import math 


class WineDataset(Dataset):

       def __init__(self):
       # data loading 
              xy = np.loadtxt("wine.csv", delimiter=",",  dtype=np.float32, skiprows=1)     
              self.x = torch.tensor(xy[:,1:], dtype = torch.float32)
              self.y = torch.tensor(xy[:, [0]], dtype= torch.float32)
              self.n_samples = xy.shape[0]

       def __getitem__(self, index):
       # dataset indexing 
              return self.x[index], self.y[index]

       def __len__(self):
              return self.n_samples


dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
dataiter = iter(dataloader)

data = next(dataiter)


features, labels =  data
print(f'feature: {features}, \nlabels: {labels}')

total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
num_epochs = 2

for epoch in range(num_epochs):
       for i, (inputs, labels) in enumerate(dataloader):
             if (i+1) % 5 == 0:
                    print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}\
                           inputs {inputs.shape}'
                          ) 