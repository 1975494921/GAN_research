from torch import nn
import torch
from torch import optim


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers1 = nn.Sequential(
            nn.Linear(10, 5, bias=False),
        )

    def forward(self, x):
        return self.layers1(x)


model = Model()
a = torch.randn(10, 10)
b = torch.randn(10, 5)
loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for param in model.parameters():
    print(param.grad)

out = model(a)
for param in model.parameters():
    print(param.grad)

loss = loss_func(out, b)
for param in model.parameters():
    print(param.grad)

loss.backward()
for param in model.parameters():
    print(param.grad)
    param.grad *= 0.5
    print(param.grad)

optimizer.zero_grad()
for param in model.parameters():
    print(param.grad)
    param.grad *= 0.5
    print(param.grad)