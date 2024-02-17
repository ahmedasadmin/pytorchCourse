import torch 
import torch.nn as nn

X = torch.tensor([1, 2, 3, 4 ], dtype=torch.float32).reshape(-1,1)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32).reshape(-1,1)
X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape
input_size = n_features
output_size = n_features


class LinearRegression(nn.Module):
       
       def __init__(self, input_dim, output_dim):
              super(LinearRegression, self).__init__()
              
              #define Layers
              self.l= nn.Linear(input_dim, output_dim)
       def forward(self, x):
              return self.l(x)


model = LinearRegression(input_size, output_size)

# w = torch.tensor(0, dtype=torch.float32, requires_grad=True)

# def forward(x):
# #     return w * x
# model = nn.Linear(input_size, output_size)
print(f'Prediction before training: f(5) = {model(X_test).item():.4f}')
learning_rate = 0.05

n_iters = 100


loss =  nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


for epoch in range(n_iters):


       y_pred = model(X)
       l = loss(Y, y_pred)
       l.backward()
       optimizer.step()
       optimizer.zero_grad()


       if epoch % 10 ==0:
             [w, b] = model.parameters()
             print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.9f}')


print(f'Prediction before training: f(5) = {model(X_test).item():.4f}')
