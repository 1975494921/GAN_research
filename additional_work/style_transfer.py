"""
    This is the additional works for the project "Art Investigate with Generative Adversarial Network" from Junting Li.
    University College London, 2023.

    Code Reference:
    [1] Neural transfer using pytorch. Neural Transfer Using PyTorch - PyTorch Tutorials 2.0.0+cu117 documentation. (n.d.). Retrieved March 26, 2023, from https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
"""
from functools import reduce
from torchvision import transforms
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

unloader = transforms.ToPILImage()
transform = transforms.ToTensor()


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.cal_gram_matrix(target_feature).detach()

    def cal_gram_matrix(self, data: torch.Tensor) -> torch.Tensor:
        dim = data.shape
        n = reduce(lambda x, y: x * y, dim)
        features = data.view(dim[0] * dim[1], dim[2] * dim[3])
        Gram_M = torch.mm(features, features.t()).div(n)

        return Gram_M

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        G = self.cal_gram_matrix(data)
        return F.mse_loss(G, self.target)


class ContentLoss(nn.Module):
    def __init__(self, target_feature):
        super(ContentLoss, self).__init__()
        self.target = target_feature.clone().detach()

    @torch.no_grad()
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(data, self.target)


class style_content_loss_module(nn.Module):
    def __init__(self, content_img, style_img, device):
        super(style_content_loss_module, self).__init__()
        if not os.path.isfile('model.pth'):
            raise FileNotFoundError('model.pth not found')

        self.model = torch.load('model.pth').to(device)
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.content_layers = ['conv_4']
        self.style_losses_calculator = dict()
        self.content_losses_calculator = dict()
        self.device = device
        self.to(device)
        self.init_features(content_img, style_img)

    def image_norm(self, img):
        mean = torch.tensor([0.485, 0.456, 0.406]).detach().view(-1, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).detach().view(-1, 1, 1).to(self.device)
        img = (img - mean) / std
        return img

    @torch.no_grad()
    def init_features(self, content_img, style_img):
        content_img = self.image_norm(content_img)
        style_img = self.image_norm(style_img)
        for name, layer in self.model.named_children():
            content_img = layer(content_img)
            style_img = layer(style_img)
            if name in self.content_layers:
                target_feature = content_img.detach()
                self.content_losses_calculator[name] = ContentLoss(target_feature)

            if name in self.style_layers:
                target_feature = style_img.detach()
                self.style_losses_calculator[name] = StyleLoss(target_feature)

    def forward(self, data):
        content_loss = 0
        style_loss = 0
        data = self.image_norm(data)
        for name, layer in self.model.named_children():
            data = layer(data)
            if name in self.content_layers:
                content_loss += self.content_losses_calculator[name](data)

            if name in self.style_layers:
                style_loss += self.style_losses_calculator[name](data)

        return content_loss, style_loss


class style_transfer:
    def __init__(self, content_img, style_img):
        self.content_img = transform(content_img).unsqueeze(0)
        self.style_img = transform(style_img).unsqueeze(0)
        self.style_weight = 1000000
        self.content_weight = 1
        self.input_img = None
        self.optimizer = None
        self.loss_model = None
        self.iteration = 0

    def image_optim(self):
        self.input_img.data.clamp_(min=0, max=1)
        self.optimizer.zero_grad()
        content_loss, style_loss = self.loss_model(self.input_img)
        style_score = self.style_weight * style_loss
        content_score = self.content_weight * content_loss
        loss = style_score + content_score
        loss.backward()

        self.iteration += 1
        if self.iteration % 50 == 0:
            print("Iter : {}:".format(self.iteration))
            print('Style Loss : {:4f} Content Loss: {:4f}\n'.format(
                style_score.item(), content_score.item()))

        return style_score + content_score

    def run(self, num_steps, device):
        self.iteration = 0
        content_img = self.content_img.to(device)
        style_img = self.style_img.to(device)
        self.input_img = content_img.clone().requires_grad_(True).to(device)
        self.optimizer = optim.LBFGS([self.input_img])

        print('Loading model......')
        self.loss_model = style_content_loss_module(content_img, style_img, device)

        print('Processing......')
        while self.iteration <= num_steps:
            self.optimizer.step(self.image_optim)

        self.input_img.data.clamp_(min=0, max=1)
        output_image = unloader(self.input_img.detach().cpu().squeeze(0)).convert('RGB')

        return np.asarray(output_image)
