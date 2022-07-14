import torch
from torch import nn
from random import randint


class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(2):
            new_layer = nn.Sequential(
                nn.Linear(1, 5),
                nn.ReLU()
            )
            self.layers.append(new_layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class model2(nn.Module):
    def __init__(self):
        super().__init__()
        new_module = model()
        self.modules = nn.ModuleList([new_module])
        new_module2 = model()
        self.modules.append(model())

a = model2()
print(a)
