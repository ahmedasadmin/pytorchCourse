import torch 
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape


X_train, X_test, y_train, y_test = train_test_split(X, y)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1)
X_test= torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32)).reshape(-1, 1)


class LogisticRegression(nn.Module):
    def __init__(self,n_input_features, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(n_input_features, 1)
    

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted



model = LogisticRegression(n_features)

criterion = nn.BCELoss()
optimizer =torch.optim.Adam(model.parameters(), lr=0.005)


epochs = 100
for _epochs in range(epochs):
    y_predicted = model(X_train)
    loss =  criterion(y_predicted, y_train)


    loss.backward()
    optimizer.step()

    optimizer.zero_grad()

    if (_epochs + 1) % 10 == 0:
        print(f'epoch: {_epochs + 1}, loss={loss.item():.5f}')


with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_classes = y_predicted.round()
    acc = y_predicted_classes.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.5f}')

