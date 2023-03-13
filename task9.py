import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.optim import Adam, SGD


device_ids = [0, 1, 2, 3]
device = torch.device("cuda:{}".format(device_ids[0]))

# set seed for reproducibility
SEED = 1855403
np.random.seed(SEED)
torch.manual_seed(SEED)

train_set = torchvision.datasets.CIFAR10(
    root="./",
    download=True,
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
)
test_set = torchvision.datasets.CIFAR10(
    root="./",
    download=True,
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
)
print('Train_set Length: ', len(train_set))
print('Test_set Length: ', len(test_set))


class res_block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.layer = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.dim, self.dim),
        )

    def forward(self, x):
        return x + self.layer(x)

class Net(nn.Module):
    def __init__(self, dim, nclass, width, depth):
        super().__init__()
        self.dim = dim
        self.nclass = nclass
        self.width = width
        self.depth = depth
        self.layers = nn.ModuleList()

        self.out_layer = nn.Sequential(
            nn.Linear(self.width, self.nclass),
            nn.Softmax(dim=1),
        )
        self._make_layer()
        self.weights_init()

    def _make_layer(self):
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(self.dim, self.width))
        self.layers.append(nn.ReLU())
        for _ in range(self.depth - 1):
            self.layers.append(res_block(self.width))
            self.layers.append(nn.ReLU())

        self.layers.append(self.out_layer)

    def weights_init(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def loading_data(batch_size, train_set, test_set):
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
    )
    return train_loader, test_loader


def train_epoch(trainloader, net, optimizer, criterion):
    net.train()
    loss_list = []
    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

    return np.mean(loss_list)


def test_epoch(testloader, net, criterion):
    net.eval()
    loss_list = []
    correct = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

    return np.mean(loss_list), 1 - correct / len(testloader.dataset)


def train(dim, width, depth, nclass, batch_size, lr, n_epoch):
    trainloader, testloader = loading_data(batch_size, train_set, test_set)
    net = Net(dim, nclass, width, depth).to(device)
    net = nn.DataParallel(net, device_ids=device_ids)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=lr)
    for epo in range(n_epoch):
        train_loss = train_epoch(trainloader, net, optimizer, criterion)
        test_loss, test_err = test_epoch(testloader, net, criterion)
        print(f'Epoch: {epo} | Train Loss: {train_loss:.04} | Test Loss: {test_loss:.04} | Test Error: {test_err:.04}')


batch_size = 64
dim = 3072
width = 128
depth = 3
nclass = 10
lr = 0.0005

if __name__ == "__main__":
    train(dim=dim, width=width, depth=depth, nclass=nclass, batch_size=batch_size, lr=lr, n_epoch=100)
