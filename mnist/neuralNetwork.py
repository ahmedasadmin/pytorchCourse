import  torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import torch
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/mnist")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# hyper parameters
input_size = 784
hidden_size = 100
num_classes = 10
batch_size = 100
learning_rate = 0.001
num_epochs = 3

# for plotting pupose

train_losses = []

training_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                               transform=transforms.ToTensor(), download=False)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                                               transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, 
                                           shuffle=False)
examples = iter(train_loader)
samples, labels = next(examples)


print(samples.shape, labels.shape)

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
    plt.axis('off')

# plt.show()
img_grid = torchvision.utils.make_grid(samples)
writer.add_image('mnist_image', img_grid)

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
    
model = NeuralNet(input_size, hidden_size, num_classes)

# loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


writer.add_graph(model, samples.reshape(-1, 28*28))
# 
# sys.exit()
n_total_steps = len(train_loader)
running_loss = 0.0
running_correct = 0.0

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()


        _, predictions = torch.max(outputs.data, 1)
        running_correct += (predictions==labels).sum().item()
        if (i+1)%100 == 0:
            print(f'epoch {epoch + 1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
            writer.add_scalar('training loss', running_loss / 100, epoch* n_total_steps + i)
            writer.add_scalar('accuracy ', running_correct / 100, epoch* n_total_steps + i)
            running_loss = 0.0
            running_correct = 0

class_labels = []
class_preds = []

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        values, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        class_probs_batch = [F.softmax(output, dim=0) for output in outputs]

        class_preds.append(class_probs_batch)
        class_labels.append(labels)

    # 10000, 10, and 10000, 1
    # stack concatenates tensors along a new dimension
    # cat concatenates tensors in the given dimension
    class_preds = torch.cat([torch.stack(batch) for batch in class_preds])
    class_labels = torch.cat(class_labels)

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')

    ############## TENSORBOARD ########################
    classes = range(10)
    for i in classes:
        labels_i = class_labels == i
        preds_i = class_preds[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
        writer.close()
    ###################################################



plt.figure(figsize=(9,5))
plt.title("Training Loss")
plt.plot(train_losses,label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()