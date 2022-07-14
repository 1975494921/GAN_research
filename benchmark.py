import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters and DataLoaders
input_size = 2000
output_size = 2000

batch_size = 5000
data_size = 100000

device_group_1 = [0, 1, 2, 3]
device_group_2 = [1, 0, 2, 3]

device1 = "cuda:{}".format(device_group_1[0])
device2 = "cuda:{}".format(device_group_2[0])


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

print("1")


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


print("2")
model1 = Model(input_size, output_size)
model2 = Model(output_size, output_size)
print("3")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model1 = nn.DataParallel(model1, device_ids=device_group_1)
    model2 = nn.DataParallel(model2, device_ids=device_group_2)

print("4")
model1.to(device1)
model2.to(device2)
print("5")
optim1 = torch.optim.Adam(params=model1.parameters(), lr=0.001)
optim2 = torch.optim.SGD(params=model2.parameters(), lr=0.001)
print("6")
loss_func = nn.MSELoss()

input = torch.randn((batch_size, input_size)).to(device1)
target = torch.randn((batch_size, output_size)).to(device2)
print("7")
print("Start training...")
while True:
    output = model1(input)
    output_final = model2(output)
    optim1.zero_grad()
    optim2.zero_grad()
    loss = loss_func(output_final, target)
    loss.backward()
    optim2.step()
    optim1.step()
