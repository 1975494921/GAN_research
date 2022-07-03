import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters and DataLoaders
input_size = 3000
output_size = 10000

batch_size = 100000
data_size = 10000000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, output_size)

    def forward(self, input):
        output = self.fc2(self.fc1(input))
        output = self.fc3(output)
        return output


model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

model.to(device)
optim = torch.optim.Adam(params=model.parameters(), lr=0.001)
loss_func = nn.MSELoss()

input = torch.randn((batch_size, input_size)).to(device)
target = torch.randn((batch_size, output_size)).to(device)
while True:
    output = model(input)
    optim.zero_grad()
    loss = loss_func(output, target)
    loss.backward()
    optim.step()
