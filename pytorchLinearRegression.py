
import torch 
import torch.nn as nn
import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt

X_test = torch.tensor([5], dtype=torch.float32)
n_iters = 100
X_train_numpy, y_train_numpy = datasets.make_regression(
    n_samples=100, n_features=1, noise=20, random_state=0
)
X_train = torch.from_numpy(X_train_numpy.astype(np.float32)).reshape(-1, 1)
y_train = torch.from_numpy(y_train_numpy.astype(np.float32)).reshape(-1, 1)

n_samples, n_features = X_train.shape
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)


Criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)



for epoch in range(n_iters):
    y_predicted = model(X_train)
    loss = Criterion(y_predicted, y_train)
    loss.backward()


    #update 'model.parameters'
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f'epoch: {epoch + 1}, loss= {loss.item():.5f}')
        
print(f'Prediction after training: model([5]) = {model(X_test).item():.7f}')


#plotting 
predicted = model(X_train).detach().numpy()


plt.plot(X_train_numpy, y_train_numpy, c="dimgray", linestyle=' ' ,marker='o', markerfacecolor='blue')
plt.plot(X_train_numpy, predicted, c='darkblue')
plt.show()